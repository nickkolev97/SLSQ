import os
import numpy as np
import cv2
import json
import torch
import argparse
import traceback
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from ultralytics import YOLO
from icecream import ic


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train YOLOv8 model for donut detection')
    
    # Add debug flag
    parser.add_argument('--debug', action='store_true', 
                        help='Run in debug mode with reduced dataset and epochs')
    
    # Model configuration
    parser.add_argument('--model_size', type=str, default='m', choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size: n=nano, s=small, m=medium, l=large, x=xlarge')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--img_size', type=int, default=640,
                        help='Input image size for training')
    
    # skip creating the dataset.yaml file if already exists. Provide path instead
    parser.add_argument('--skip_yaml_creation', type=str, default=None,
                        help='Skip dataset conversion if dataset.yaml already exists')

    # Data paths
    parser.add_argument('--data_dir', type=str, default='synthetic_data_3',
                        help='Path to the dataset directory')
    
    return parser.parse_args()


def convert_to_yolo_format(images, annotations, output_dir='yolo_dataset'):
    """Convert your synthetic dataset to YOLO format."""
    
    # Create directory structure
    os.makedirs(f"{output_dir}/images/train", exist_ok=True)
    os.makedirs(f"{output_dir}/images/val", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/val", exist_ok=True)
    
    print(f"Converting dataset to YOLO format in {output_dir}...")
    
    # Determine train/val split
    num_images = len(images)
    val_size = int(num_images * 0.2)  # 20% for validation
    indices = np.arange(num_images)
    np.random.shuffle(indices)
    val_indices = indices[:val_size]
    
    # Process images
    for i in tqdm(range(num_images)):
        img = images[i]
        ann = annotations[i]
        
        # Determine split
        split = "val" if i in val_indices else "train"
        
        # max/min normalization
        img = (img - img.min()) / (img.max() - img.min())

        # Save image (convert to RGB if it's grayscale)
        if len(img.shape) == 2 or img.shape[2] == 1:
            img_rgb = np.stack([img] * 3, axis=2) if len(img.shape) == 2 else np.repeat(img, 3, axis=2)
        else:
            img_rgb = img
            
        img_path = f"{output_dir}/images/{split}/{i:06d}.png"
        cv2.imwrite(img_path, (img_rgb * 255).astype(np.uint8))
        
        # Create YOLO format labels
        h, w = img.shape[:2]
        with open(f"{output_dir}/labels/{split}/{i:06d}.txt", 'w') as f:
            if 'boxes' in ann:
                for box in ann['boxes']:
                    # Convert to YOLO format: class_id, x_center, y_center, width, height (all normalized)
                    x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
                    x_center = box['center_x'] / w
                    y_center = box['center_y'] / h
                    box_width = (x2 - x1) / w
                    box_height = (y2 - y1) / h
                    # if x1, y1, x2, y2, x_center, y_center are over 1, it means the box is out of the image. Change them to 1
                    if x1 > 1:
                        x1 = 1
                    if y1 > 1:
                        y1 = 1
                    if x2 > 1:
                        x2 = 1
                    if y2 > 1:
                        y2 = 1
                    if x_center > 1:
                        x_center = 1
                    if y_center > 1:
                        y_center = 1
                    f.write(f"0 {x_center} {y_center} {box_width} {box_height}\n")
                  #  ic(f"{x_center} {y_center} {box_width} {box_height}")
    
    # Create dataset.yaml file
    with open(f"{output_dir}/dataset.yaml", 'w') as f:
        f.write(f"path: {Path(output_dir).absolute()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/val\n")
        f.write("nc: 1\n")  # number of classes
        f.write("names:\n  0: donut\n")
    
    print(f"Dataset conversion complete. Created YAML config at {output_dir}/dataset.yaml")
    return f"{output_dir}/dataset.yaml"


def load_dataset(dir, debug=False):
    """Load the synthetic donut dataset.
    args:
        dir: str, path to the synthetic_data directory
        debug: bool, if True, only load a small subset of data
    returns:
        train_images: np.array, training images
        train_annotations: list, training annotations
        val_images: np.array, validation images
        val_annotations: list, validation annotations
    """
    print("Loading dataset...")
    
    try:
        # Load training data
        train_images = np.load(f'{dir}/numpy_arrays/train_images.npy')
        with open(f'{dir}/annotations/train_annotations.json', 'r') as f:
            train_annotations = json.load(f)
            
        # Load validation data
        val_images = np.load(f'{dir}/numpy_arrays/val_images.npy')
        with open(f'{dir}/annotations/val_annotations.json', 'r') as f:
            val_annotations = json.load(f)
        
        # If in debug mode, use only a small subset of the data
        if debug:
            # Use only 20 training images and 10 validation images for debugging
            train_limit = min(500, len(train_images))
            val_limit = min(100, len(val_images))
            
            train_images = train_images[:train_limit]
            train_annotations = train_annotations[:train_limit]
            val_images = val_images[:val_limit]
            val_annotations = val_annotations[:val_limit]
            
            print(f"DEBUG MODE: Using {train_limit} training and {val_limit} validation images")
        else:
            print(f"Loaded {len(train_images)} training and {len(val_images)} validation images")
            
        return train_images, train_annotations, val_images, val_annotations
    
    except FileNotFoundError as e:
        print(f"Dataset files not found: {e}")
        print("Please make sure the synthetic_data directory exists.")
        return None, None, None, None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(traceback.format_exc())
        return None, None, None, None


def train_yolov8(dataset_yaml, model_size='n', epochs=50, img_size=640, batch_size=16, debug=False):
    """Train YOLOv8 model on the dataset."""
    
    # Set model size ('n', 's', 'm', 'l', 'x')
    model_options = {
        'n': 'yolov8n.pt',  # Nano
        's': 'yolov8s.pt',  # Small
        'm': 'yolov8m.pt',  # Medium
        'l': 'yolov8l.pt',  # Large
        'x': 'yolov8x.pt'   # Extra-large
    }

    model_path = model_options.get(model_size, 'yolov8n.pt')
    
    # Create output directories
    os.makedirs('models/yolov8', exist_ok=True)
    
    # If in debug mode, reduce epochs
    if debug:
        epochs = min(2, epochs)
        print(f"DEBUG MODE: Training YOLOv8-{model_size} for {epochs} epochs")
    else:
        print(f"Training YOLOv8-{model_size} for {epochs} epochs")
    
    try:
        # Initialize model
        model = YOLO('yolov8m-dropout.yaml')
        model.load('yolov8m.pt')
    
        # Train the model
        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            project='models',
            name=f'yolov8_{model_size}_donuts',
            exist_ok=True,
            save=True,
            workers=1
        )

        return model, f'models/yolov8_{model_size}_donuts'
        
    except Exception as e:
        print(f"Error during training: {e}")
        print(traceback.format_exc())
        return None, None
    

def evaluate_and_visualize(model, val_images, val_annotations, num_samples=5):
    """Evaluate the model and visualize predictions on validation samples."""
    
    if model is None:
        print("Cannot evaluate: model is None")
        return
        
    print("Evaluating and visualizing results...")
    
    try:
        # Select random samples
        indices = np.random.choice(len(val_images), min(num_samples, len(val_images)), replace=False)
        
        plt.figure(figsize=(15, 5 * num_samples))
        
        for i, idx in enumerate(indices):
            img = val_images[idx]
            ann = val_annotations[idx]
            
            # Convert image to suitable format
            if len(img.shape) == 2:
                img_display = np.repeat(img[:, :, np.newaxis], 3, axis=2)
            else:
                img_display = img
                
            img_display = (img_display * 255).astype(np.uint8)
            
            # Save image temporarily for prediction
            temp_img_path = "temp_prediction.png"
            cv2.imwrite(temp_img_path, img_display)
            
            # Run prediction
            results = model.predict(temp_img_path, conf=0.25)
            #boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format

            # Ground truth image
            gt_img = img_display.copy()
            
            for box in ann['boxes']:
                x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
                cv2.rectangle(gt_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Plot ground truth and predictions side by side
            plt.subplot(num_samples, 2, 2*i+1)
            plt.title(f"Sample {idx} - Ground Truth")
            plt.imshow(gt_img)
            plt.axis('off')
            
            # Plot predictions
            plt.subplot(num_samples, 2, 2*i+2)
            plt.title(f"Sample {idx} - YOLOv8 Predictions")
            plt.imshow(results[0].plot(conf=False, labels=False, line_width=1))
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('models/yolov8_results.png')
        
        # Don't call plt.show() on a headless server
        if 'DISPLAY' in os.environ:
            plt.show()
        
        # Clean up
        if os.path.exists("temp_prediction.png"):
            os.remove("temp_prediction.png")
            
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print(traceback.format_exc())


def main():
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Print debug status
        if args.debug:
            print("Running in DEBUG mode: reduced dataset and epochs")
        
        # Load the dataset
        train_images, train_annotations, val_images, val_annotations = load_dataset(args.data_dir, debug=args.debug)

        # makes sense up to here
        
        if train_images is None:
            return
        
        if args.skip_yaml_creation is None:
            # Convert dataset to YOLO format
            dataset_yaml = convert_to_yolo_format(
                np.concatenate((train_images, val_images)),
                train_annotations + val_annotations
            )
        else:
            dataset_yaml = args.skip_yaml_creation

        # Train YOLOv8 model
        model, output_dir = train_yolov8(
            dataset_yaml,
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            debug=args.debug
        )
        
        # Evaluate and visualize results
        if model is not None:
            evaluate_and_visualize(model, val_images, val_annotations)
            print(f"Training complete. Model saved to {output_dir}")
        else:
            print("Training failed, no model to evaluate.")
            
    except Exception as e:
        print(f"Error in main function: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()