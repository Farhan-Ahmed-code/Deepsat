import os
import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

# Suppress TensorFlow warnings for clean CLI output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

# Configuration and normalization parameters
NORM_MEAN = [0.3484, 0.3824, 0.4085]
NORM_STD = [0.2102, 0.1413, 0.1186]

def load_label_map(label_map_path):
    if not os.path.exists(label_map_path):
        print(f"[-] Label map file not found at: {label_map_path}")
        return None
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    # Return class names sorted by their index mapping
    return sorted(list(label_map.keys()), key=lambda x: label_map[x])

def preprocess_image(image_path):
    """Loads and preprocesses a single image to match EuroSAT baseline model input shape and normalization."""
    if not os.path.exists(image_path):
        print(f"[-] Error: Image file not found at: {image_path}")
        sys.exit(1)
    
    try:
        # Load and convert to RGB
        img = Image.open(image_path).convert('RGB')
        # Resize to 64x64 (EuroSAT native size)
        img = img.resize((64, 64))
        # Scale to [0, 1]
        img_arr = np.array(img, dtype=np.float32) / 255.0
        
        # Apply normalization: (x - mean) / std
        mean = np.array(NORM_MEAN, dtype=np.float32)
        std = np.array(NORM_STD, dtype=np.float32)
        img_arr = (img_arr - mean) / std
        
        # Add batch dimension: (1, 64, 64, 3)
        img_arr = np.expand_dims(img_arr, axis=0)
        return img_arr
    except Exception as e:
        print(f"[-] Error preprocessing image {image_path}: {e}")
        sys.exit(1)

def print_separator(char="=", length=70):
    print(char * length)

def display_single_prediction(class_names, predictions, image_name):
    # predictions shape: (1, 10)
    probs = predictions[0]
    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]
    
    print("\n" + "=" * 70)
    print(f"  PREDICTION RESULT FOR: {image_name}")
    print("=" * 70)
    print(f"  --> Predicted Class:  \033[1;32m{class_names[pred_idx]}\033[0m")
    print(f"  --> Confidence:       \033[1;36m{confidence:.2%}\033[0m")
    print("-" * 70)
    print("  Class Probabilities Distribution:")
    
    # Print sorted classes by probability
    sorted_indices = np.argsort(probs)[::-1]
    for idx in sorted_indices:
        prob = probs[idx]
        class_name = class_names[idx]
        # Generate a small bar chart
        bar_length = int(prob * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"    * {class_name:<22}: [{bar}] {prob:.2%}")
    print("=" * 70 + "\n")

def run_random_predictions(model, class_names, base_dir, test_csv_path, count=5):
    if not os.path.exists(test_csv_path):
        print(f"[-] Error: Test metadata CSV not found at: {test_csv_path}")
        return
    
    df = pd.read_csv(test_csv_path)
    # Choose random rows
    sampled_df = df.sample(n=min(count, len(df)), random_state=random.randint(1, 10000))
    
    print("\n" + "=" * 70)
    print(f"          RUNNING INFERENCE ON {len(sampled_df)} RANDOM TEST IMAGES")
    print("=" * 70)
    
    headers = f"{'Image Filename':<30} | {'True Class':<18} | {'Predicted Class':<18} | {'Conf':<7}"
    print(headers)
    print("-" * len(headers))
    
    correct_count = 0
    for idx, row in sampled_df.iterrows():
        rel_path = row['Filename']
        true_cls = row['ClassName']
        full_path = os.path.join(base_dir, rel_path)
        
        # Preprocess and predict
        img_tensor = preprocess_image(full_path)
        pred_probs = model.predict(img_tensor, verbose=0)[0]
        pred_idx = np.argmax(pred_probs)
        pred_cls = class_names[pred_idx]
        confidence = pred_probs[pred_idx]
        
        # Stylize correctly / incorrectly classified items
        if true_cls == pred_cls:
            correct_count += 1
            pred_str = f"\033[32m{pred_cls:<18}\033[0m"
        else:
            pred_str = f"\033[31m{pred_cls:<18}\033[0m"
            
        filename_short = os.path.basename(rel_path)
        print(f"{filename_short:<30} | {true_cls:<18} | {pred_str} | {confidence:.1%}")
        
    print("-" * len(headers))
    accuracy = correct_count / len(sampled_df)
    print(f"  Batch Accuracy: {correct_count}/{len(sampled_df)} ({accuracy:.1%})")
    print("=" * 70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="EuroSAT Baseline CNN Inference / Prediction CLI Utility")
    parser.add_argument("image_path", nargs="?", default=None, help="Path to an image to classify.")
    parser.add_argument("--random", type=int, default=0, help="Run prediction on N random images from the test set.")
    args = parser.parse_args()
    
    model_path = os.path.join("models", "best_baseline_cnn.keras")
    dataset_dir = os.path.join("EuroSAT_Dataset", "2750")
    label_map_path = os.path.join(dataset_dir, "label_map.json")
    test_csv_path = os.path.join(dataset_dir, "test.csv")
    
    # 1. Load Model
    if not os.path.exists(model_path):
        print(f"[-] Error: Trained model not found at: {model_path}")
        print("    Please run train_baseline.py to train the model first.")
        sys.exit(1)
        
    print(f"[+] Loading EuroSAT baseline model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # 2. Load labels
    class_names = load_label_map(label_map_path)
    if class_names is None:
        print("[-] Error loading class names. Exiting.")
        sys.exit(1)
        
    # 3. Determine Execution Path
    if args.image_path is not None:
        # User specified a specific image
        print(f"[+] Running single-image inference on: {args.image_path}")
        img_tensor = preprocess_image(args.image_path)
        preds = model.predict(img_tensor, verbose=0)
        display_single_prediction(class_names, preds, os.path.basename(args.image_path))
    elif args.random > 0:
        # User requested random test predictions
        run_random_predictions(model, class_names, dataset_dir, test_csv_path, count=args.random)
    else:
        # Default behavior: run on 5 random images to showcase the model in action
        print("[+] No image path provided. Defaulting to 5 random test set predictions...")
        run_random_predictions(model, class_names, dataset_dir, test_csv_path, count=5)
        print("[TIP] You can also run: ")
        print("      python predict.py <path_to_custom_satellite_image_jpeg>")
        print("      python predict.py --random 10")

if __name__ == "__main__":
    main()
