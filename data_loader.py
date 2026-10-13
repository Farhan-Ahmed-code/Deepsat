import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# =====================================================================
# 1. REPRODUCIBILITY & CONFIGURATION
# =====================================================================
# Set random seeds for numpy and tensorflow to ensure reproducible splits/shuffling
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Normalization constants calculated during Exploratory Data Analysis (EDA)
# These represent the per-channel mean and standard deviation of EuroSAT
NORM_MEAN = [0.3484, 0.3824, 0.4085]
NORM_STD = [0.2102, 0.1413, 0.1186]

# =====================================================================
# 2. PREPROCESSING FUNCTIONS
# =====================================================================
def load_and_preprocess_image(file_path, label):
    """
    TensorFlow mapping function to load, decode, resize, and normalize an image.
    Works entirely with TensorFlow operations for compatibility with tf.data.Dataset.
    """
    # Read the raw file as a byte string
    img = tf.io.read_file(file_path)
    
    # Decode the byte string to a 3-channel (RGB) image tensor
    # decode_jpeg is fast and safe as EuroSAT contains standard JPEG images
    img = tf.image.decode_jpeg(img, channels=3)
    
    # Resize the image to 64x64 pixels (EuroSAT is natively 64x64, but we force this 
    # to ensure consistency in size)
    img = tf.image.resize(img, [64, 64])
    
    # Convert pixels to float32 and scale from [0, 255] to [0.0, 1.0]
    img = tf.cast(img, tf.float32) / 255.0
    
    # Normalize the image using Z-score normalization: (x - mean) / std
    mean = tf.constant(NORM_MEAN, dtype=tf.float32)
    std = tf.constant(NORM_STD, dtype=tf.float32)
    img = (img - mean) / std
    
    return img, label

def denormalize_image(img):
    """
    Helper function to reverse the normalization for visualization.
    Converts normalized float32 images back to [0.0, 1.0] range.
    """
    mean = np.array(NORM_MEAN)
    std = np.array(NORM_STD)
    
    # Reconstruct original image scale: (img * std) + mean
    img = (img * std) + mean
    
    # Clip values to ensure they lie strictly within standard [0.0, 1.0] display range
    img = np.clip(img, 0.0, 1.0)
    
    return img

# =====================================================================
# 3. DATASET GENERATION PIPELINE
# =====================================================================
def get_dataset(csv_path, dataset_dir, batch_size=32, shuffle=False, augment=False):
    """
    Creates an optimized tf.data.Dataset for the EuroSAT training, validation, or test split.
    
    Parameters:
    - csv_path (str): Path to the split CSV file (train, validation, or test).
    - dataset_dir (str): Base path to the EuroSAT images directory.
    - batch_size (int): Number of samples per batch.
    - shuffle (bool): If True, shuffles the dataset order.
    - augment (bool): If True, applies random augmentations (flips & small rotations).
    
    Returns:
    - tf.data.Dataset: Configured dataset pipeline ready for model training.
    """
    # Load the CSV file containing relative filenames and labels
    df = pd.read_csv(csv_path)
    
    # Construct complete absolute/normalized paths for each image
    file_paths = df['Filename'].apply(lambda x: os.path.normpath(os.path.join(dataset_dir, x))).tolist()
    labels = df['Label'].tolist()
    
    # Create the base dataset from the list of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    # Shuffle the dataset order if requested (crucial for training)
    # Shuffle early to avoid shuffling preprocessed/augmented data in memory
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(file_paths), seed=SEED)
    
    # Load and preprocess the images in parallel using CPU threads
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch the elements
    dataset = dataset.batch(batch_size)
    
    # Apply data augmentation (Only applied to the training dataset split)
    if augment:
        # Define spatial augmentation layers using Keras sequential
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=SEED),
            tf.keras.layers.RandomRotation(factor=0.05, fill_mode="reflect", seed=SEED)  # Small rotation (~18 deg)
        ])
        
        # Apply the Keras layers onto image batches in the pipeline
        # Keras layers automatically handle batch-wise operations efficiently
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y),
                              num_parallel_calls=tf.data.AUTOTUNE)
    
    # Prefetch dataset batches to memory to allow the GPU to train on the current batch 
    # while the CPU prepares the next batch in parallel
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

# =====================================================================
# 4. VERIFICATION AND TESTING ENTRYPOINT
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    # Define absolute/relative workspace paths
    base_dataset_dir = os.path.join("EuroSAT_Dataset", "2750")
    train_csv_path = os.path.join(base_dataset_dir, "train.csv")
    val_csv_path = os.path.join(base_dataset_dir, "validation.csv")
    test_csv_path = os.path.join(base_dataset_dir, "test.csv")
    label_map_path = os.path.join(base_dataset_dir, "label_map.json")
    
    # Check if necessary files exist
    required_paths = [train_csv_path, val_csv_path, test_csv_path, label_map_path]
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required dataset file: {path}")
    
    # Load the class labels mapping
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    # Reverse the map to lookup class names from numeric labels
    inv_label_map = {v: k for k, v in label_map.items()}
    
    print("[+] Dataset files verified successfully!")
    print(f"[+] Loaded Class Mapping: {label_map}")
    
    # Create training, validation, and test datasets
    # Batch size of 32 for standard training verification
    batch_size = 32
    print("\n[+] Creating TensorFlow dataset pipelines...")
    train_dataset = get_dataset(train_csv_path, base_dataset_dir, batch_size=batch_size, shuffle=True, augment=True)
    val_dataset = get_dataset(val_csv_path, base_dataset_dir, batch_size=batch_size, shuffle=False, augment=False)
    test_dataset = get_dataset(test_csv_path, base_dataset_dir, batch_size=batch_size, shuffle=False, augment=False)
    
    # Pull one batch from each dataset to verify shapes
    print("\n[+] Pulling verification batches...")
    for name, dataset in [("Train (with Augmentation)", train_dataset), 
                          ("Validation (No Augmentation)", val_dataset), 
                          ("Test (No Augmentation)", test_dataset)]:
        
        # Take exactly 1 batch from the dataset iterator
        for img_batch, label_batch in dataset.take(1):
            print(f"    - {name} Batch Shapes:")
            print(f"      * Image Batch: {img_batch.shape} (Expected: ({batch_size}, 64, 64, 3))")
            print(f"      * Label Batch: {label_batch.shape} (Expected: ({batch_size},))")
            
            # Basic sanity checks
            assert img_batch.shape == (batch_size, 64, 64, 3), f"Incorrect image batch shape for {name}!"
            assert label_batch.shape == (batch_size,), f"Incorrect label batch shape for {name}!"
            
            # Check value range of normalized images (should have negative and positive values)
            img_min = tf.reduce_min(img_batch).numpy()
            img_max = tf.reduce_max(img_batch).numpy()
            print(f"      * Value Range (Normalized): Min={img_min:.4f}, Max={img_max:.4f}")
            
    print("\n[+] Basic pipeline assertions passed successfully!")
    
    # =====================================================================
    # 5. VISUAL TESTING & GRID GENERATION
    # =====================================================================
    print("\n[+] Generating visual preprocessing comparison grid...")
    fig, axes = plt.subplots(2, 5, figsize=(15, 7.5))
    
    # 1. Grab sample batch from Training Dataset (Augmented & Shuffled)
    for train_imgs, train_lbls in train_dataset.take(1):
        for i in range(5):
            img_normalized = train_imgs[i].numpy()
            img_denormalized = denormalize_image(img_normalized)
            lbl = train_lbls[i].numpy()
            class_name = inv_label_map[lbl]
            
            axes[0, i].imshow(img_denormalized)
            axes[0, i].set_title(f"Train (Augmented):\n{class_name}", fontsize=11, fontweight='bold', pad=8)
            axes[0, i].axis('off')
            
    # 2. Grab sample batch from Validation Dataset (Non-augmented, sequential)
    for val_imgs, val_lbls in val_dataset.take(1):
        for i in range(5):
            img_normalized = val_imgs[i].numpy()
            img_denormalized = denormalize_image(img_normalized)
            lbl = val_lbls[i].numpy()
            class_name = inv_label_map[lbl]
            
            axes[1, i].imshow(img_denormalized)
            axes[1, i].set_title(f"Val (Original):\n{class_name}", fontsize=11, pad=8)
            axes[1, i].axis('off')
            
    plt.suptitle("EuroSAT Preprocessed Pipeline Samples\n(Row 1: Augmented Training Samples | Row 2: Standard Validation Samples)", 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save comparison grid to disk
    output_img_path = "preprocessed_samples.png"
    plt.savefig(output_img_path, dpi=150)
    plt.close()
    
    print(f"[+] Visual validation grid successfully saved to: {output_img_path}")
    print("=" * 60)
