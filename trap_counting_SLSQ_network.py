print('Loading modules...')
import matplotlib.pyplot as plt
import numpy as np
import patchify as pat
import cv2
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead
from torch.utils.data import Dataset, DataLoader
import json
import os
from tqdm import tqdm

print('Defining classes and functions...')

# Custom dataset class for loading the synthetic data
class DonutDataset(Dataset):
    def __init__(self, images_file, annotations_file, transform=None, device = 'cpu'):
        self.images = np.load(images_file)
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image and convert to RGB (3 channels)
        img = self.images[idx]
        img = np.stack([img, img, img], axis=2)  # Convert grayscale to RGB
        
        # Get annotations for this image
        ann = next((a for a in self.annotations if a['image_id'] == idx), None)

        # Extract bounding box coordinates from potentially nested structure
        box_list = []
        for box in ann['boxes']:
            if isinstance(box, dict):
                # If each box is a dictionary, extract coordinates
                # This assumes each dict has x1, y1, x2, y2 or similar keys
                # Adjust these keys based on your actual data structure
                if all(k in box for k in ['x1', 'y1', 'x2', 'y2']):
                    box_list.append([box['x1'], box['y1'], box['x2'], box['y2']])
                # Alternative format with 'x', 'y', 'width', 'height'
                elif all(k in box for k in ['x', 'y', 'width', 'height']):
                    box_list.append([box['x'], box['y'], box['x'] + box['width'], box['y'] + box['height']])
            else:
                # If it's already a list, use it directly
                box_list.append(box)
        
        boxes = torch.tensor(box_list, dtype=torch.float32)
        
        # Create target dictionary
        target = {}
        target['boxes'] = boxes
        target['labels'] = torch.ones((boxes.shape[0],), dtype=torch.int64)  # All boxes are donuts (class 1)
        
        # Convert image to tensor
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        if self.transform:
            img = self.transform(img)
        
        return img, target, ann

# Function to convert dataset annotations to target tensors
def create_targets(images, annotations, output_size=(16, 16)):
    """
    Convert annotations to target tensors for training
    
    Args:
        images: Batch of images [B, C, H, W]
        annotations: List of dicts with 'boxes' key
        output_size: Size of the output feature map
    
    Returns:
        Dict with 'confidence' and 'boxes' tensors
    """
    batch_size = len(images)
    device = images.device
    
    # Initialize targets
    target_conf = torch.zeros(batch_size, 1, output_size[0], output_size[1], device=device)
    target_boxes = torch.zeros(batch_size, 4, output_size[0], output_size[1], device=device)
    
    # Get input image dimensions
    _, _, img_h, img_w = images.shape
    
    # Scale factors
    scale_x = output_size[1] / img_w
    scale_y = output_size[0] / img_h
    
   
    for b, ann in enumerate(annotations):
        # For each bounding box in the image
        for box in ann['boxes']:
            x1, y1, x2, y2 = box
            
            # Convert to center format
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            
            # Scale to feature map coordinates
            cx_scaled = cx * scale_x
            cy_scaled = cy * scale_y
            
            # Get the grid cell where center falls
            grid_x = int(cx_scaled)
            grid_y = int(cy_scaled)
            
            # Only proceed if within bounds
            if 0 <= grid_x < output_size[1] and 0 <= grid_y < output_size[0]:
                # Set confidence to 1 at this location
                target_conf[b, 0, grid_y, grid_x] = 1.0
                
                # Set bounding box parameters (normalized to cell)
                target_boxes[b, 0, grid_y, grid_x] = cx_scaled - grid_x  # x offset in cell (0-1)
                target_boxes[b, 1, grid_y, grid_x] = cy_scaled - grid_y  # y offset in cell (0-1)
                target_boxes[b, 2, grid_y, grid_x] = w * scale_x / output_size[1]  # width relative to image
                target_boxes[b, 3, grid_y, grid_x] = h * scale_y / output_size[0]  # height relative to image
    
    return {'confidence': target_conf, 'boxes': target_boxes}

# Function for post-processing predictions
def decode_predictions(predictions, threshold=0.5, input_size=(512, 512)):
    """
    Decode network predictions to get bounding boxes
    
    Args:
        predictions: Dict with 'confidence' and 'boxes' from network
        threshold: Confidence threshold
        input_size: Original image size (height, width)
    
    Returns:
        List of dicts with 'boxes' and 'scores' for each image
    """
    batch_size = predictions['confidence'].shape[0]
    conf = torch.sigmoid(predictions['confidence'])
    boxes = predictions['boxes']
    
    # Get feature map size
    _, _, feat_h, feat_w = conf.shape
    
    # Scale factors to convert back to image coordinates
    scale_x = input_size[1] / feat_w
    scale_y = input_size[0] / feat_h
    
    results = []
    for b in range(batch_size):
        # Find locations above threshold
        locations = (conf[b, 0] > threshold).nonzero(as_tuple=True)
        grid_y, grid_x = locations
        
        # Get scores and box parameters for these locations
        scores = conf[b, 0, grid_y, grid_x]
        
        # If no detections, return empty results
        if len(scores) == 0:
            results.append({'boxes': torch.zeros((0, 4)), 'scores': torch.zeros(0)})
            continue
        
        # Get box parameters
        x_offset = boxes[b, 0, grid_y, grid_x]
        y_offset = boxes[b, 1, grid_y, grid_x]
        rel_width = boxes[b, 2, grid_y, grid_x]
        rel_height = boxes[b, 3, grid_y, grid_x]
        
        # Convert to image coordinates
        cx = (grid_x + x_offset) * scale_x
        cy = (grid_y + y_offset) * scale_y
        width = rel_width * input_size[1]
        height = rel_height * input_size[0]
        
        # Convert to [x1, y1, x2, y2] format
        x1 = cx - width / 2
        y1 = cy - height / 2
        x2 = cx + width / 2
        y2 = cy + height / 2
        
        # Clip to image boundaries
        x1 = torch.clamp(x1, 0, input_size[1]-1)
        y1 = torch.clamp(y1, 0, input_size[0]-1)
        x2 = torch.clamp(x2, 0, input_size[1]-1)
        y2 = torch.clamp(y2, 0, input_size[0]-1)
        
        # Stack into boxes tensor
        boxes_out = torch.stack([x1, y1, x2, y2], dim=1)
        
        results.append({'boxes': boxes_out, 'scores': scores})
    
    return results

# detector network
class MultiDonutDetector(nn.Module):
    def __init__(self, num_classes=1):
        super(MultiDonutDetector, self).__init__()
        
        # Use MobileNetV2 as the backbone (lightweight for CPU training)
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        
        # Extract intermediate features at different scales
        self.layer1 = backbone[:2]   # 1/2 resolution
        self.layer2 = backbone[2:4]  # 1/4 resolution
        self.layer3 = backbone[4:7]  # 1/8 resolution
        self.layer4 = backbone[7:14] # 1/16 resolution
        self.layer5 = backbone[14:]  # 1/32 resolution
        
        # Prediction heads - keeping the network simple
        self.conf_head = nn.Conv2d(1280, 1, kernel_size=1)  # Object confidence
        self.bbox_head = nn.Conv2d(1280, 4, kernel_size=1)  # bounding box (x,y,w,h)
        
    def forward(self, x):
        # Feature extraction
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        features = self.layer5(x4)
        
        # Make predictions
        confidence = self.conf_head(features)
        bboxes = self.bbox_head(features)
        
        return {
            'confidence': confidence,  # Shape: [batch, 1, H/32, W/32]
            'boxes': bboxes            # Shape: [batch, 4, H/32, W/32]
        }

# Custom loss function
class MultiDonutLoss(nn.Module):
    def __init__(self, lambda_conf=1.0, lambda_bbox=1.0):
        super(MultiDonutLoss, self).__init__()
        self.lambda_conf = lambda_conf
        self.lambda_bbox = lambda_bbox
        
    def forward(self, predictions, targets):
        pred_conf = predictions['confidence']
        pred_boxes = predictions['boxes']
        
        target_conf = targets['confidence']  # [batch, 1, H/32, W/32]
        target_boxes = targets['boxes']      # [batch, 4, H/32, W/32] 
        
        # Binary cross entropy for confidence
        conf_loss = F.binary_cross_entropy_with_logits(pred_conf, target_conf)
        
        # L1 loss for bounding boxes (only where objects exist)
        # We first create a mask from target_conf to focus on cells with objects
        pos_mask = (target_conf > 0.5).float()
        bbox_loss = F.smooth_l1_loss(pred_boxes * pos_mask.repeat(1, 4, 1, 1), 
                                    target_boxes * pos_mask.repeat(1, 4, 1, 1),
                                    reduction='sum')
        
        # Normalize by number of positive cells
        num_pos = torch.max(pos_mask.sum(), torch.tensor(1.0).to(pos_mask.device))
        bbox_loss = bbox_loss / num_pos
        
        # Total loss
        total_loss = self.lambda_conf * conf_loss + self.lambda_bbox * bbox_loss
        
        return total_loss, {'conf_loss': conf_loss, 'bbox_loss': bbox_loss}

# train function
def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=10, save_path='models_for_SLSQ/donut_detector_best.pth'):
    model.to(device)
    
    loss_history = []
    val_loss_history = []
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            # Unpack the batch
            images, target, annotations = batch
            
            # For nested tuple structures, handle differently
            if isinstance(images, tuple):
                images = torch.stack(images)
            
            # Move data to device
            images = images.to(device)
            
            # Create target tensors
            targets = create_targets(images, target, output_size=(16, 16))
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(images)
            
            # Calculate loss
            loss, loss_components = criterion(predictions, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Calculate average training loss for this epoch
        avg_train_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                # Unpack the batch
                images, target, annotations = batch
                
                if isinstance(images, tuple):
                    images = torch.stack(images)
                
                # Move data to device
                images = images.to(device)
                
                # Create target tensors
                targets = create_targets(images, target, output_size=(16, 16))
                
                # Forward pass
                predictions = model(images)
                
                # Calculate loss
                loss, _ = criterion(predictions, targets)
                val_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save model if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Validation loss improved to {avg_val_loss:.4f}. Model saved to '{save_path}'")
    
    print(f"Best validation loss: {best_val_loss:.4f}")
    return loss_history, val_loss_history

# Function to evaluate the model
def evaluate_model(model, data_loader, device, num_preds=10):
    model.eval()
    
    with torch.no_grad():
        # Choose a random batch to visualize
        batch_idx = random.randint(0, len(data_loader)-1)

        for batch in tqdm(data_loader):
            # Unpack the batch - this is the critical fix
            images, target, annotations = batch
                   
            # For nested tuple structures, we need to handle differently
            if isinstance(images, tuple):
                # Convert tuple of images to tensor batch
                images = torch.stack(images)
            
            # Move data to device
            images = images.to(device)

            # Forward pass
            predictions = model(images)
            
            # decode predictions
            decoded_predictions = decode_predictions(predictions)
            
            # visualise prediction from n outputs
            for i in range(num_preds):
                visualise(images[i], decoded_predictions[i])
            break

def visualise(img, ann, prediction=False):
    '''
    Visualise an img+annotation.
    If prediction = False, then the annotations are ground truth. Used only for labelling.
    '''
    

    plt.figure(figsize=(10,10))
    plt.imshow(img[0,:,:], cmap='gray')
    
    if prediction:
        plt.title('Ground truths')
    else:
        plt.title('Model prediction')

    donut_boxes = ann['boxes']
    # Draw bounding boxes
    for box in donut_boxes:
        # Create rectangle patch
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
        width = x2 - x1
        height = y2 - y1
        
        # Draw rectangle
        rect = plt.Rectangle((x1, y1), width, height, 
                            fill=False, edgecolor='red', linewidth=1)
        plt.gca().add_patch(rect)
        
        # Optionally, draw center point
        plt.plot(box['center_x'], box['center_y'], 'r+')

    plt.title(f'Synthetic Image with Donut Bounding Boxes ({len(donut_boxes)} donuts)')
    plt.axis('off')
    plt.tight_layout()
    #plt.show()
    plt.savefig('models_for_SLSQ/eval_example.png')


    # Print bounding box information for first 5 donuts
    for i, box in enumerate(donut_boxes[:5]):
        print(f"Donut {i+1}:")
        print(f"  Bounding box: ({box['x1']}, {box['y1']}) to ({box['x2']}, {box['y2']})")
        print(f"  Center: ({box['center_x']}, {box['center_y']})")
        print(f"  Outer radius: {box['outer_radius']}, Inner radius: {box['inner_radius']}")


if __name__ == '__main__':
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print('Creating datasets and dataloaders...')
    # Create datasets and dataloaders
    train_dataset = DonutDataset(
        'synthetic_data/numpy_arrays/train_images.npy',
        'synthetic_data/annotations/train_annotations.json'
    )

    val_dataset = DonutDataset(
        'synthetic_data/numpy_arrays/val_images.npy',
        'synthetic_data/annotations/val_annotations.json'
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=lambda batch: tuple(zip(*batch)))  # Required for detection models
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda batch: tuple(zip(*batch)))
    
    print('Initialising model...')
    # Initialize the model
    model = MultiDonutDetector()
    Loss = MultiDonutLoss()
        
    # Set up optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Train the model
    loss_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=Loss,
        optimizer=optimizer,
        device=device,
        num_epochs=15
    )

    # Save the model
    torch.save(model.state_dict(), 'models_for_SLSQ/donut_detector_final.pth')
    print("Model saved to 'models_for_SLSQ/donut_detector_final.pth'")


    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('models_for_SLSQ/training_loss.png')
    #plt.show()

    # Evaluate on validation set
    print("Evaluating model on validation set...")
    evaluate_model(model, val_loader, device)

