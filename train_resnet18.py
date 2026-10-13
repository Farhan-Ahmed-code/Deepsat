import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Import custom project modules
from data_loader import get_dataset
from resnet18 import build_resnet18_model

# =====================================================================
# 1. REPRODUCIBILITY CONFIGURATION
# =====================================================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =====================================================================
# 2. DIRECTORIES SETUP
# =====================================================================
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# =====================================================================
# 3. DATASET LOADING
# =====================================================================
print("=" * 60)
print("             EUROSAT RESNET-18 MODEL - TRAINING")
print("=" * 60)

base_dataset_dir = os.path.join("EuroSAT_Dataset", "2750")
train_csv_path = os.path.join(base_dataset_dir, "train.csv")
val_csv_path = os.path.join(base_dataset_dir, "validation.csv")

print("[+] Loading train and validation datasets...")
batch_size = 32
train_dataset = get_dataset(train_csv_path, base_dataset_dir, batch_size=batch_size, shuffle=True, augment=True)
val_dataset = get_dataset(val_csv_path, base_dataset_dir, batch_size=batch_size, shuffle=False, augment=False)

# =====================================================================
# 4. MODEL BUILDING & COMPILATION
# =====================================================================
print("\n[+] Building the ResNet-18 Model...")
model = build_resnet18_model(input_shape=(64, 64, 3), num_classes=10)

print("[+] Compiling the model...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

# Extract parameters count
total_params = model.count_params()
trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])

# =====================================================================
# 5. CALLBACKS CONFIGURATION
# =====================================================================
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join("models", "best_resnet18.keras"),
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

callbacks = [checkpoint_cb, early_stopping_cb, reduce_lr_cb]

# =====================================================================
# 6. MODEL TRAINING
# =====================================================================
max_epochs = 20
print(f"\n[+] Starting training for up to {max_epochs} epochs...")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=max_epochs,
    callbacks=callbacks,
    verbose=1
)

# =====================================================================
# 7. SAVE TRAINING HISTORY & LOGS
# =====================================================================
history_dict = {key: [float(val) for val in values] for key, values in history.history.items()}
history_json_path = os.path.join("models", "history_resnet18.json")

print(f"\n[+] Saving training history to: {history_json_path}")
with open(history_json_path, "w") as f:
    json.dump(history_dict, f, indent=4)

# =====================================================================
# 8. TRAINING PLOTS GENERATION
# =====================================================================
print("[+] Generating and saving accuracy/loss visualization curves...")
epochs_completed = len(history_dict["loss"])
epochs_range = range(1, epochs_completed + 1)

# Plot 1: Accuracy Curve
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, history_dict["accuracy"], label="Training Accuracy", marker="o", color="#1f77b4")
plt.plot(epochs_range, history_dict["val_accuracy"], label="Validation Accuracy", marker="s", color="#ff7f0e")
plt.title("EuroSAT ResNet-18: Accuracy Curve", fontsize=14, fontweight="bold", pad=12)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=11)
plt.xticks(epochs_range)
plt.tight_layout()
accuracy_plot_path = os.path.join("plots", "accuracy_resnet18.png")
plt.savefig(accuracy_plot_path, dpi=150)
plt.close()
print(f"    - Accuracy plot saved to: {accuracy_plot_path}")

# Plot 2: Loss Curve
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, history_dict["loss"], label="Training Loss", marker="o", color="#d62728")
plt.plot(epochs_range, history_dict["val_loss"], label="Validation Loss", marker="s", color="#2ca02c")
plt.title("EuroSAT ResNet-18: Loss Curve", fontsize=14, fontweight="bold", pad=12)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=11)
plt.xticks(epochs_range)
plt.tight_layout()
loss_plot_path = os.path.join("plots", "loss_resnet18.png")
plt.savefig(loss_plot_path, dpi=150)
plt.close()
print(f"    - Loss plot saved to: {loss_plot_path}")

# =====================================================================
# 9. OUTPUT STATISTICS
# =====================================================================
best_epoch_idx = np.argmin(history_dict["val_loss"])
best_epoch = best_epoch_idx + 1

best_train_acc = history_dict["accuracy"][best_epoch_idx]
best_val_acc = history_dict["val_accuracy"][best_epoch_idx]
best_val_loss = history_dict["val_loss"][best_epoch_idx]

print("\n" + "=" * 60)
print("             TRAINING SUMMARY COMPLETE")
print("=" * 60)
print(f"[*] ResNet-18 Parameters count:")
print(f"    - Total Parameters:     {total_params:,}")
print(f"    - Trainable Parameters: {trainable_params:,}")
print(f"    - Non-trainable Params: {non_trainable_params:,}")
print(f"[*] Epochs completed:      {epochs_completed} / {max_epochs}")
print(f"[*] Best epoch selected:   Epoch {best_epoch} (based on validation loss)")
print(f"[*] Training Accuracy:     {best_train_acc:.4f}")
print(f"[*] Validation Accuracy:   {best_val_acc:.4f}")
print(f"[*] Validation Loss:       {best_val_loss:.4f}")
print(f"[*] Saved Model location:  models/best_resnet18.keras")
print(f"[*] Saved Plots location:  plots/accuracy_resnet18.png, plots/loss_resnet18.png")
print("=" * 60)
