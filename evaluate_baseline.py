import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from data_loader import get_dataset

# =====================================================================
# 1. DIRECTORIES & PATHS CONFIGURATION
# =====================================================================
os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

base_dataset_dir = os.path.join("EuroSAT_Dataset", "2750")
test_csv_path = os.path.join(base_dataset_dir, "test.csv")
label_map_path = os.path.join(base_dataset_dir, "label_map.json")
model_path = os.path.join("models", "best_baseline_cnn.keras")

# =====================================================================
# 2. LOAD DATA AND MODEL
# =====================================================================
print("=" * 60)
print("             EUROSAT BASELINE CNN - TEST EVALUATION")
print("=" * 60)

# Load class label mapping
print("[+] Loading class mapping and test metadata...")
with open(label_map_path, "r") as f:
    label_map = json.load(f)
class_names = sorted(list(label_map.keys()), key=lambda x: label_map[x])
inv_label_map = {v: k for k, v in label_map.items()}

# Load test dataset via pandas to track absolute file paths
test_df = pd.read_csv(test_csv_path)
true_labels = test_df["Label"].values
filenames = test_df["Filename"].values

# Load the trained model checkpoint
print(f"[+] Loading trained model from: {model_path}")
model = tf.keras.models.load_model(model_path)

# Build TensorFlow dataset pipeline for prediction
print("[+] Building TF evaluation dataset...")
batch_size = 32
test_dataset = get_dataset(test_csv_path, base_dataset_dir, batch_size=batch_size, shuffle=False, augment=False)

# =====================================================================
# 3. PREDICTIONS INFERENCE
# =====================================================================
print("[+] Running model predictions on the TEST set (2,700 images)...")
predictions = model.predict(test_dataset, verbose=1)

# Extract predicted class indices and confidences
pred_classes = np.argmax(predictions, axis=1)
pred_confidences = np.max(predictions, axis=1)

# =====================================================================
# 4. METRIC COMPUTATION
# =====================================================================
print("[+] Computing classification metrics...")

# Calculate overall and macro scores
test_acc = accuracy_score(true_labels, pred_classes)
macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(true_labels, pred_classes, average="macro")

# Calculate per-class metrics
precisions, recalls, f1_scores, supports = precision_recall_fscore_support(true_labels, pred_classes, average=None)

# Compile per-class metrics into dictionary
per_class_metrics = {}
for i, name in enumerate(class_names):
    per_class_metrics[name] = {
        "precision": float(precisions[i]),
        "recall": float(recalls[i]),
        "f1-score": float(f1_scores[i]),
        "support": int(supports[i])
    }

# Save overall results to JSON
results_dict = {
    "test_accuracy": float(test_acc),
    "macro_precision": float(macro_prec),
    "macro_recall": float(macro_rec),
    "macro_f1": float(macro_f1),
    "per_class_metrics": per_class_metrics
}

results_json_path = os.path.join("results", "baseline_cnn_test_results.json")
with open(results_json_path, "w") as f:
    json.dump(results_dict, f, indent=4)
print(f"    - Metrics JSON saved to: {results_json_path}")

# =====================================================================
# 5. EXPORT CLASSIFICATION REPORT CSV
# =====================================================================
# Generate standard scikit-learn classification report dictionary
report_dict = classification_report(true_labels, pred_classes, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

report_csv_path = os.path.join("results", "baseline_cnn_classification_report.csv")
report_df.to_csv(report_csv_path)
print(f"    - Classification report CSV saved to: {report_csv_path}")

# =====================================================================
# 6. CONFUSION MATRIX GENERATION
# =====================================================================
print("[+] Generating confusion matrix visualization...")
cm = confusion_matrix(true_labels, pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    cmap="Blues", 
    xticklabels=class_names, 
    yticklabels=class_names,
    cbar=True,
    square=True
)
plt.title("EuroSAT Baseline CNN: Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
plt.ylabel("True Label", fontsize=12, fontweight="bold")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

cm_plot_path = os.path.join("plots", "baseline_cnn_confusion_matrix.png")
plt.savefig(cm_plot_path, dpi=150)
plt.close()
print(f"    - Confusion matrix plot saved to: {cm_plot_path}")

# =====================================================================
# 7. MISCLASSIFIED EXAMPLES PLOT
# =====================================================================
print("[+] Finding and plotting misclassified examples...")
# Find indices where prediction does not match ground truth label
misclassified_indices = np.where(pred_classes != true_labels)[0]
print(f"    - Found {len(misclassified_indices)} misclassified images in test set.")

# Select a subset of exactly 10 images to plot (using a deterministic skip to sample across classes)
# Take 10 spread out indices to get variety
step = max(1, len(misclassified_indices) // 10)
selected_indices = [misclassified_indices[i * step] for i in range(10)]

fig, axes = plt.subplots(2, 5, figsize=(15, 7.5))
axes = axes.flatten()

for idx, test_idx in enumerate(selected_indices):
    rel_path = filenames[test_idx]
    full_path = os.path.join(base_dataset_dir, rel_path)
    
    true_cls = inv_label_map[true_labels[test_idx]]
    pred_cls = inv_label_map[pred_classes[test_idx]]
    confidence = pred_confidences[test_idx]
    
    # Load original raw JPEG directly for high-quality visual reference
    img = Image.open(full_path)
    
    axes[idx].imshow(img)
    axes[idx].set_title(
        f"True: {true_cls}\nPred: {pred_cls}\nConf: {confidence:.2%}", 
        fontsize=10, 
        fontweight="bold",
        color="red",
        pad=6
    )
    axes[idx].axis("off")

plt.suptitle("EuroSAT Baseline CNN: Representative Misclassified Test Examples", fontsize=15, fontweight="bold", y=0.98)
plt.tight_layout()

misclassified_plot_path = os.path.join("results", "misclassified_examples.png")
plt.savefig(misclassified_plot_path, dpi=150)
plt.close()
print(f"    - Misclassified examples plot saved to: {misclassified_plot_path}")

# =====================================================================
# 8. PERFORMANCE INSIGHTS AND SUMMARY
# =====================================================================
# Sort classes by F1-score to find best and worst performers
sorted_classes = sorted(class_names, key=lambda name: per_class_metrics[name]["f1-score"])
worst_classes = sorted_classes[:2]
best_classes = sorted_classes[-2:]

print("\n" + "=" * 60)
print("             EVALUATION SUMMARY COMPLETE")
print("=" * 60)
print(f"[*] Test Accuracy:    {test_acc:.4f}")
print(f"[*] Macro Precision:  {macro_prec:.4f}")
print(f"[*] Macro Recall:     {macro_rec:.4f}")
print(f"[*] Macro F1-score:   {macro_f1:.4f}")
print(f"[*] Best Performing Classes:  {best_classes[-1]} ({per_class_metrics[best_classes[-1]]['f1-score']:.4f}), {best_classes[-2]} ({per_class_metrics[best_classes[-2]]['f1-score']:.4f})")
print(f"[*] Worst Performing Classes: {worst_classes[0]} ({per_class_metrics[worst_classes[0]]['f1-score']:.4f}), {worst_classes[1]} ({per_class_metrics[worst_classes[1]]['f1-score']:.4f})")
print("=" * 60)
