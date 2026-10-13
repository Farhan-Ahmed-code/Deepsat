import os
import numpy as np
import tensorflow as tf
from data_loader import get_dataset

# =====================================================================
# 1. REPRODUCIBILITY CONFIGURATION
# =====================================================================
# Set fixed seeds to ensure reproducible model initialization and batching
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =====================================================================
# 2. MODEL ARCHITECTURE DEFINITION
# =====================================================================
def build_baseline_model(input_shape=(64, 64, 3), num_classes=10):
    """
    Builds the baseline CNN model from scratch for EuroSAT land cover classification.
    
    Architecture details:
    - Input: 64x64 RGB images
    - Conv2D (32 filters, 3x3) -> BatchNormalization -> ReLU -> MaxPool (2x2)
    - Conv2D (64 filters, 3x3) -> BatchNormalization -> ReLU -> MaxPool (2x2)
    - Conv2D (128 filters, 3x3) -> BatchNormalization -> ReLU -> MaxPool (2x2)
    - GlobalAveragePooling2D (flattens spatial dimensions into a feature vector)
    - Fully Connected Dense layer (128 units, ReLU activation)
    - Dropout (30% regularization rate)
    - Output Dense layer (10 units, Softmax activation)
    """
    model = tf.keras.Sequential([
        # Input definition
        tf.keras.layers.Input(shape=input_shape),
        
        # Block 1
        tf.keras.layers.Conv2D(32, (3, 3), padding="same", name="conv_block1_conv"),
        tf.keras.layers.BatchNormalization(name="conv_block1_bn"),
        tf.keras.layers.ReLU(name="conv_block1_relu"),
        tf.keras.layers.MaxPooling2D((2, 2), name="conv_block1_pool"),
        
        # Block 2
        tf.keras.layers.Conv2D(64, (3, 3), padding="same", name="conv_block2_conv"),
        tf.keras.layers.BatchNormalization(name="conv_block2_bn"),
        tf.keras.layers.ReLU(name="conv_block2_relu"),
        tf.keras.layers.MaxPooling2D((2, 2), name="conv_block2_pool"),
        
        # Block 3
        tf.keras.layers.Conv2D(128, (3, 3), padding="same", name="conv_block3_conv"),
        tf.keras.layers.BatchNormalization(name="conv_block3_bn"),
        tf.keras.layers.ReLU(name="conv_block3_relu"),
        tf.keras.layers.MaxPooling2D((2, 2), name="conv_block3_pool"),
        
        # Classifier Head
        tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool"),
        tf.keras.layers.Dense(128, activation="relu", name="fc_dense"),
        tf.keras.layers.Dropout(0.3, name="fc_dropout"),
        tf.keras.layers.Dense(num_classes, activation="softmax", name="output_softmax")
    ], name="EuroSAT_Baseline_CNN")
    
    return model

# =====================================================================
# 3. VERIFICATION AND DRY-RUN
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("      EUROSAT BASELINE CNN MODEL - INITIALIZATION & TESTING")
    print("=" * 60)
    
    # Define dataset directory paths
    base_dataset_dir = os.path.join("EuroSAT_Dataset", "2750")
    train_csv_path = os.path.join(base_dataset_dir, "train.csv")
    val_csv_path = os.path.join(base_dataset_dir, "validation.csv")
    test_csv_path = os.path.join(base_dataset_dir, "test.csv")
    
    print("[+] Loading dataset pipeline to fetch test batches...")
    batch_size = 32
    train_dataset = get_dataset(train_csv_path, base_dataset_dir, batch_size=batch_size, shuffle=True, augment=True)
    val_dataset = get_dataset(val_csv_path, base_dataset_dir, batch_size=batch_size, shuffle=False, augment=False)
    test_dataset = get_dataset(test_csv_path, base_dataset_dir, batch_size=batch_size, shuffle=False, augment=False)
    
    # Instantiate the baseline model
    print("[+] Building baseline CNN model...")
    model = build_baseline_model(input_shape=(64, 64, 3), num_classes=10)
    
    # Display the model summary and parameters
    print("\n[+] Model Structure Summary:")
    model.summary()
    
    # Compile the model as specified: Adam optimizer, Sparse Categorical Crossentropy, Accuracy metric
    print("\n[+] Compiling model...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    print("    - Optimizer: Adam (lr=0.001)")
    print("    - Loss: SparseCategoricalCrossentropy")
    print("    - Metrics: ['accuracy']")
    
    # Test forward and backward pass using exactly 1 training batch
    print("\n[+] Verifying pipeline execution with a single-batch test run...")
    for images, labels in train_dataset.take(1):
        # 1. Forward Pass Test (inference mode)
        print("    - Performing forward pass test...")
        predictions = model(images, training=False)
        print(f"      * Inputs shape:       {images.shape}")
        print(f"      * Predictions shape:  {predictions.shape} (Expected: ({batch_size}, 10))")
        assert predictions.shape == (batch_size, 10), "Prediction shape mismatch!"
        
        # 2. Backward Pass / Training Step Test
        print("    - Performing backward pass / training step test...")
        metrics = model.train_on_batch(images, labels, return_dict=True)
        print(f"      * Loss:     {metrics['loss']:.4f}")
        print(f"      * Accuracy: {metrics['accuracy']:.4f}")
        
    print("\n[+] Baseline CNN compile and single-batch execution successful!")
    print("=" * 60)
