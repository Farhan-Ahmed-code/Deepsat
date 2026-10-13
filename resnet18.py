import os
import numpy as np
import tensorflow as tf
from data_loader import get_dataset

# =====================================================================
# 1. REPRODUCIBILITY CONFIGURATION
# =====================================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =====================================================================
# 2. RESNET-18 COMPONENTS
# =====================================================================
def residual_block(x, filters, strides=1, downsample=False, name_prefix=""):
    """
    Standard ResNet basic residual block with two 3x3 convolutions and a shortcut.
    """
    shortcut = x
    
    # If spatial dimensions change (strides > 1) or channel count changes, adjust shortcut
    if downsample:
        shortcut = tf.keras.layers.Conv2D(
            filters, (1, 1), strides=strides, padding="same", use_bias=False,
            name=f"{name_prefix}_shortcut_conv"
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_shortcut_bn")(shortcut)
        
    # Conv 1
    x = tf.keras.layers.Conv2D(
        filters, (3, 3), strides=strides, padding="same", use_bias=False,
        name=f"{name_prefix}_conv1"
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = tf.keras.layers.ReLU(name=f"{name_prefix}_relu1")(x)
    
    # Conv 2
    x = tf.keras.layers.Conv2D(
        filters, (3, 3), strides=1, padding="same", use_bias=False,
        name=f"{name_prefix}_conv2"
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    
    # Residual addition and activation
    x = tf.keras.layers.Add(name=f"{name_prefix}_add")([x, shortcut])
    x = tf.keras.layers.ReLU(name=f"{name_prefix}_relu2")(x)
    return x

# =====================================================================
# 3. MODEL ARCHITECTURE DEFINITION
# =====================================================================
def build_resnet18_model(input_shape=(64, 64, 3), num_classes=10):
    """
    Builds the ResNet-18 model for EuroSAT land cover classification.
    
    Architecture details:
    - Input: 64x64 RGB images
    - Stage 0 (Stem): Conv2D 7x7 (64 filters, stride 2) -> BN -> ReLU -> MaxPooling2D (stride 2)
    - Stage 1: 2 basic blocks (64 filters, stride 1)
    - Stage 2: 2 basic blocks (128 filters, stride 2)
    - Stage 3: 2 basic blocks (256 filters, stride 2)
    - Stage 4: 2 basic blocks (512 filters, stride 2)
    - Head: GlobalAveragePooling2D -> Dense (num_classes, Softmax)
    """
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_image")
    
    # Stem: 7x7 Conv -> BN -> ReLU -> MaxPool
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=2, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = tf.keras.layers.BatchNormalization(name="stem_bn")(x)
    x = tf.keras.layers.ReLU(name="stem_relu")(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding="same", name="stem_pool")(x)
    
    # Stage 1: 64 filters, stride 1 (output remains 16x16)
    x = residual_block(x, 64, strides=1, downsample=False, name_prefix="stage1_block1")
    x = residual_block(x, 64, strides=1, downsample=False, name_prefix="stage1_block2")
    
    # Stage 2: 128 filters, stride 2 (output size: 8x8)
    x = residual_block(x, 128, strides=2, downsample=True, name_prefix="stage2_block1")
    x = residual_block(x, 128, strides=1, downsample=False, name_prefix="stage2_block2")
    
    # Stage 3: 256 filters, stride 2 (output size: 4x4)
    x = residual_block(x, 256, strides=2, downsample=True, name_prefix="stage3_block1")
    x = residual_block(x, 256, strides=1, downsample=False, name_prefix="stage3_block2")
    
    # Stage 4: 512 filters, stride 2 (output size: 2x2)
    x = residual_block(x, 512, strides=2, downsample=True, name_prefix="stage4_block1")
    x = residual_block(x, 512, strides=1, downsample=False, name_prefix="stage4_block2")
    
    # Classifier Head
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="output_softmax")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="EuroSAT_ResNet18")
    return model

# =====================================================================
# 4. VERIFICATION AND DRY-RUN
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("      EUROSAT RESNET-18 MODEL - INITIALIZATION & TESTING")
    print("=" * 60)
    
    # Define dataset directory paths
    base_dataset_dir = os.path.join("EuroSAT_Dataset", "2750")
    train_csv_path = os.path.join(base_dataset_dir, "train.csv")
    
    print("[+] Loading training dataset pipeline to fetch test batches...")
    batch_size = 32
    train_dataset = get_dataset(train_csv_path, base_dataset_dir, batch_size=batch_size, shuffle=True, augment=True)
    
    # Instantiate the ResNet-18 model
    print("[+] Building ResNet-18 model...")
    model = build_resnet18_model(input_shape=(64, 64, 3), num_classes=10)
    
    # Display the model summary and parameters
    print("\n[+] Model Structure Summary:")
    model.summary()
    
    # Compile the model
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
        
    print("\n[+] ResNet-18 compile and single-batch execution successful!")
    print("=" * 60)
