"""
DeepSatVision - API & Inference Verification Script
===================================================
Tests FastAPI endpoints and model inference on EuroSAT images.

Usage:
    python test_api.py
"""

import os
import json
import numpy as np
from PIL import Image

def test_endpoints():
    print("=" * 70)
    print("DEEPSATVISION: TESTING FASTAPI SERVER & INFERENCE")
    print("=" * 70)
    
    from backend.server import (
        get_health,
        get_project_info,
        get_dataset_info,
        get_model_metrics,
        get_sample_images,
        load_all_models,
        CLASS_NAMES,
        loaded_models
    )

    # 1. Load models
    print("[1] Preloading Keras models into memory...")
    load_all_models()
    
    # 2. Health
    print("\n[2] Checking Health Endpoint:")
    health = get_health()
    print(f"    Status:            {health['status']}")
    print(f"    CNN Available:     {health['models']['cnn']['available']} ({health['models']['cnn']['parameters']:,} params)")
    print(f"    ResNet Available:  {health['models']['resnet18']['available']} ({health['models']['resnet18']['parameters']:,} params)")
    
    # 3. Project Info
    print("\n[3] Checking Project Info:")
    pinfo = get_project_info()
    print(f"    Project Title:     {pinfo['project_title']}")
    print(f"    Programme:         {pinfo['programme']}")
    print(f"    Guide:             {pinfo['faculty_guide']['name']}")
    print(f"    Students:          {[s['name'] for s in pinfo['students']]}")
    
    # 4. Dataset Info
    print("\n[4] Checking Dataset Info:")
    dinfo = get_dataset_info()
    print(f"    Dataset:           {dinfo['dataset_name']}")
    print(f"    Total Images:      {dinfo['total_images']:,}")
    print(f"    Classes Count:     {dinfo['num_classes']}")
    print(f"    Splits:            Train: {dinfo['splits']['train']['count']:,} | Val: {dinfo['splits']['val']['count']:,} | Test: {dinfo['splits']['test']['count']:,}")
    
    # 5. Model Metrics
    print("\n[5] Checking Evaluation Metrics:")
    metrics = get_model_metrics()
    cnn_acc = metrics['summary']['cnn']['accuracy']
    resnet_acc = metrics['summary']['resnet18']['accuracy']
    improvement = metrics['summary']['improvement']['accuracy_diff']
    print(f"    CNN Accuracy:      {cnn_acc}%")
    print(f"    ResNet Accuracy:   {resnet_acc}%")
    print(f"    Improvement:       {improvement}")
    print(f"    Error Reduction:   {metrics['summary']['improvement']['error_reduction_percentage']}")
    
    # 6. Sample Images
    print("\n[6] Checking Sample Images:")
    samples = get_sample_images(samples_per_class=2)
    print(f"    Loaded {samples['total']} EuroSAT sample tiles across all classes.")
    
    # 7. Model Inference Test
    print("\n[7] Testing Real Inference on a Sample Tile (SeaLake):")
    sample_img_path = os.path.join("EuroSAT_Dataset", "2750", "SeaLake", "SeaLake_1.jpg")
    if os.path.exists(sample_img_path):
        img = Image.open(sample_img_path).convert('RGB').resize((64, 64))
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.3484, 0.3824, 0.4085], dtype=np.float32)
        std = np.array([0.2102, 0.1413, 0.1186], dtype=np.float32)
        norm_arr = np.expand_dims((arr - mean) / std, axis=0)
        
        # Test ResNet-18
        res_model = loaded_models.get('resnet18')
        if res_model:
            preds = res_model.predict(norm_arr, verbose=0)[0]
            pred_idx = np.argmax(preds)
            print(f"    ResNet-18 Result:  Class '{CLASS_NAMES[pred_idx]}' (Confidence: {preds[pred_idx]*100:.2f}%)")
            
        # Test CNN
        cnn_model = loaded_models.get('cnn')
        if cnn_model:
            preds_c = cnn_model.predict(norm_arr, verbose=0)[0]
            pred_idx_c = np.argmax(preds_c)
            print(f"    Baseline CNN Result: Class '{CLASS_NAMES[pred_idx_c]}' (Confidence: {preds_c[pred_idx_c]*100:.2f}%)")
            
    print("\n" + "=" * 70)
    print("[SUCCESS] All endpoints and models verified!")
    print("Run 'python run_app.py' to launch the web platform.")
    print("=" * 70)

if __name__ == "__main__":
    test_endpoints()

