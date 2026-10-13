import os
import json
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Import dataset generation logic
from data_loader import get_dataset

def main():
    # 1. Setup paths
    base_dataset_dir = os.path.join("EuroSAT_Dataset", "2750")
    test_csv_path = os.path.join(base_dataset_dir, "test.csv")
    label_map_path = os.path.join(base_dataset_dir, "label_map.json")
    
    models_dir = "models"
    cnn_model_path = os.path.join(models_dir, "best_baseline_cnn.keras")
    resnet_model_path = os.path.join(models_dir, "best_resnet18.keras")
    
    results_dir = "results"
    plots_dir = "plots"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # 2. Load dataset mappings
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    
    # We will reverse mapping first
    inv_label_map = {v: k for k, v in label_map.items()}
    class_names = [inv_label_map[i] for i in range(len(inv_label_map))]

    # 3. Load dataset
    batch_size = 32
    print("Loading test dataset...")
    test_dataset = get_dataset(test_csv_path, base_dataset_dir, batch_size=batch_size, shuffle=False, augment=False)

    # Pre-extract true labels for sklearn metrics
    print("Extracting true labels...")
    y_true = []
    for _, y in test_dataset:
        y_true.extend(y.numpy())
    y_true = np.array(y_true)

    # Function to evaluate a model
    def evaluate_model(model_path, model_name):
        print(f"\nEvaluating {model_name}...")
        model = tf.keras.models.load_model(model_path)
        
        # Test loss and accuracy
        loss, accuracy = model.evaluate(test_dataset, verbose=0)
        
        # Predict
        y_pred_probs = model.predict(test_dataset, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        
        # Classification report
        report_str = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        
        file_suffix = model_name.lower().replace('-', '')
        with open(os.path.join(results_dir, f"classification_report_{file_suffix}.txt"), 'w') as f:
            f.write(report_str)
            
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix: {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"confusion_matrix_{file_suffix}.png"), dpi=150)
        plt.close()
        
        return {
            'Model': model_name,
            'Test Accuracy': accuracy * 100,
            'Test Loss': loss,
            'Precision': precision * 100,
            'Recall': recall * 100,
            'F1 Score': f1 * 100
        }

    # 4. Evaluate models
    cnn_results = evaluate_model(cnn_model_path, "CNN")
    resnet_results = evaluate_model(resnet_model_path, "ResNet-18")
    
    # 5. Generate comparison
    results_list = [cnn_results, resnet_results]
    df_results = pd.DataFrame(results_list)
    df_results.to_csv(os.path.join(results_dir, "final_test_results.csv"), index=False)
    
    improvement = resnet_results['Test Accuracy'] - cnn_results['Test Accuracy']

    # 6. Print summary
    print("\nFINAL TEST RESULTS")
    print("-" * 18)
    
    print("Baseline CNN:")
    print(f"Test Accuracy: {cnn_results['Test Accuracy']:.2f}%")
    print(f"Precision: {cnn_results['Precision']:.2f}%")
    print(f"Recall: {cnn_results['Recall']:.2f}%")
    print(f"F1 Score: {cnn_results['F1 Score']:.2f}%")
    print()
    print("ResNet-18:")
    print(f"Test Accuracy: {resnet_results['Test Accuracy']:.2f}%")
    print(f"Precision: {resnet_results['Precision']:.2f}%")
    print(f"Recall: {resnet_results['Recall']:.2f}%")
    print(f"F1 Score: {resnet_results['F1 Score']:.2f}%")
    print()
    print("Improvement:")
    print(f"ResNet-18 - CNN = {improvement:.2f} percentage points")

if __name__ == "__main__":
    main()
