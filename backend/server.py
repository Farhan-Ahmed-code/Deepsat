import os
import io
import time
import json
import logging
import numpy as np
import pandas as pd
from PIL import Image
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# Suppress TensorFlow verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATASET_DIR = os.path.join(BASE_DIR, "EuroSAT_Dataset", "2750")

# EuroSAT channel-wise normalization parameters
NORM_MEAN = [0.3484, 0.3824, 0.4085]
NORM_STD = [0.2102, 0.1413, 0.1186]

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

# In-memory model cache
loaded_models: Dict[str, Any] = {}
loaded_metadata: Dict[str, Any] = {}

def load_all_models():
    """Pre-load Keras models into memory for ultra-low latency live inference."""
    cnn_path = os.path.join(MODELS_DIR, "best_baseline_cnn.keras")
    resnet_path = os.path.join(MODELS_DIR, "best_resnet18.keras")
    
    if os.path.exists(cnn_path):
        try:
            print(f"[Backend] Loading Baseline CNN from {cnn_path}...")
            loaded_models["cnn"] = tf.keras.models.load_model(cnn_path)
            # Warm up forward pass
            dummy = np.zeros((1, 64, 64, 3), dtype=np.float32)
            _ = loaded_models["cnn"].predict(dummy, verbose=0)
            print("[Backend] Baseline CNN loaded and warmed up.")
        except Exception as e:
            print(f"[Backend ERROR] Failed to load Baseline CNN: {e}")
            loaded_models["cnn"] = None
    else:
        loaded_models["cnn"] = None

    if os.path.exists(resnet_path):
        try:
            print(f"[Backend] Loading ResNet-18 from {resnet_path}...")
            loaded_models["resnet18"] = tf.keras.models.load_model(resnet_path)
            dummy = np.zeros((1, 64, 64, 3), dtype=np.float32)
            _ = loaded_models["resnet18"].predict(dummy, verbose=0)
            print("[Backend] ResNet-18 loaded and warmed up.")
        except Exception as e:
            print(f"[Backend ERROR] Failed to load ResNet-18: {e}")
            loaded_models["resnet18"] = None
    else:
        loaded_models["resnet18"] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models into memory
    print("[Backend] Starting DeepSatVision AI API Server...")
    load_all_models()
    yield
    # Shutdown
    print("[Backend] Shutting down DeepSatVision API Server...")

app = FastAPI(
    title="DeepSatVision API",
    description="AI-Based Land Use Classification from Sentinel-2 Satellite Images (B.Tech Minor Project)",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for React frontend (Vite default ports 5173, 3000, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folders for plots and EuroSAT dataset images
if os.path.exists(PLOTS_DIR):
    app.mount("/static/plots", StaticFiles(directory=PLOTS_DIR), name="plots")
if os.path.exists(DATASET_DIR):
    app.mount("/static/dataset", StaticFiles(directory=DATASET_DIR), name="dataset")
if os.path.exists(os.path.join(BASE_DIR, "results")):
    app.mount("/static/results", StaticFiles(directory=RESULTS_DIR), name="results")

def preprocess_pil_image(img: Image.Image) -> np.ndarray:
    """Preprocess PIL Image matching EuroSAT training pipeline."""
    img_rgb = img.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
    img_arr = np.array(img_rgb, dtype=np.float32) / 255.0
    
    mean = np.array(NORM_MEAN, dtype=np.float32)
    std = np.array(NORM_STD, dtype=np.float32)
    img_normalized = (img_arr - mean) / std
    return np.expand_dims(img_normalized, axis=0)

# =====================================================================
# API Endpoints
# =====================================================================

@app.get("/api/health")
def get_health():
    return {
        "status": "healthy",
        "service": "DeepSatVision AI API",
        "timestamp": time.time(),
        "models": {
            "cnn": {
                "available": loaded_models.get("cnn") is not None,
                "parameters": 111946 if loaded_models.get("cnn") else None
            },
            "resnet18": {
                "available": loaded_models.get("resnet18") is not None,
                "parameters": 11191242 if loaded_models.get("resnet18") else None
            }
        }
    }

@app.get("/api/project-info")
def get_project_info():
    return {
        "project_title": "DeepSatVision: AI-Based Land Use Classification from Sentinel-2 Images",
        "course": "Minor Project (ETMN100)",
        "programme": "B.Tech Computer Science and Engineering (Data Science)",
        "semester": "VII",
        "institution": {
            "university": "Amity University Uttar Pradesh, Noida",
            "school": "Amity School of Engineering and Technology",
            "department": "Department of Computer Science and Engineering"
        },
        "students": [
            {
                "name": "Farhan Ahmed Siddique",
                "role": "Student Researcher / ML Engineer",
                "programme": "B.Tech CSE (Data Science) - VII Sem"
            },
            {
                "name": "Adesh Pal",
                "role": "Student Researcher / Data Analyst",
                "programme": "B.Tech CSE (Data Science) - VII Sem"
            }
        ],
        "faculty_guide": {
            "name": "Dr Aakanshi Gupta",
            "designation": "Assistant Professor-III",
            "department": "Department of Computer Science and Engineering",
            "institution": "Amity University Uttar Pradesh, Noida"
        },
        "abstract": (
            "DeepSatVision presents an empirical deep learning computer vision framework for automated Land Use and "
            "Land Cover (LULC) classification using multi-spectral Sentinel-2 satellite imagery from the benchmark EuroSAT "
            "RGB dataset (27,000 images across 10 classes). The study implements and rigorously evaluates two deep learning "
            "architectures: a custom-designed Baseline Convolutional Neural Network (111.9K parameters) and a full ResNet-18 "
            "deep residual network (11.19M parameters) engineered from scratch without pretrained weights. Systematic "
            "evaluation proves that ResNet-18 outperforms the baseline CNN across every quantitative evaluation metric, "
            "achieving 94.04% test accuracy (a +2.93% improvement) with a 32.9% reduction in misclassification errors. "
            "The solution is integrated into a high-performance web dashboard engineered for live interactive inference."
        ),
        "problem_statement": (
            "Rapid urbanization, agricultural shifts, and climate impacts demand precise, scalable, and automated "
            "environmental monitoring. Manual ground surveys and traditional remote sensing techniques are resource-intensive, "
            "slow, and subjective. Deep learning on Sentinel-2 satellite imagery provides an automated, scalable solution, "
            "yet requires comparative benchmarking to validate whether deep residual architectures provide tangible gains "
            "over lightweight convolutional networks on low-resolution 64x64 multi-spectral scenes."
        ),
        "objectives": [
            "Implement a robust, scalable data loading and augmentation pipeline using TensorFlow tf.data for the EuroSAT RGB dataset.",
            "Design and train a custom lightweight Convolutional Neural Network (CNN) tailored for 64x64 satellite patches.",
            "Implement a complete 18-layer Residual Network (ResNet-18) from scratch with identity and projection shortcut blocks.",
            "Train both architectures from scratch under identical hyperparameter conditions to ensure scientific comparability.",
            "Conduct an in-depth quantitative benchmark analyzing test accuracy, macro precision, recall, F1-score, and confusion matrices.",
            "Deliver an interactive full-stack research platform for live single-image and batch satellite image inference."
        ],
        "methodology_steps": [
            {"step": 1, "title": "Dataset Acquisition", "desc": "EuroSAT RGB dataset consisting of 27,000 Sentinel-2 images downloaded and verified."},
            {"step": 2, "title": "Exploratory Data Analysis", "desc": "Computed channel-wise statistics (RGB mean & std), inspected class distribution and image integrity."},
            {"step": 3, "title": "Stratified Splitting", "desc": "Stratified division into 70% Training (18,900), 20% Validation (5,400), and 10% Testing (2,700)."},
            {"step": 4, "title": "Preprocessing & Z-Score Scaling", "desc": "Channel-wise zero-mean standardization: (x - mean) / std based on dataset-wide metrics."},
            {"step": 5, "title": "Training-Only Augmentation", "desc": "Random horizontal/vertical flips and ±18° rotations to prevent overfitting on satellite orientations."},
            {"step": 6, "title": "tf.data Optimization Pipeline", "desc": "Parallel mapping, batching (batch size 32/64), buffer shuffling, and AUTOTUNE prefetching."},
            {"step": 7, "title": "Custom Baseline CNN Construction", "desc": "3 Conv blocks (32, 64, 128 filters) + BatchNormalization + MaxPool + GAP + Dense(128) + Dropout(0.3)."},
            {"step": 8, "title": "ResNet-18 Scratch Construction", "desc": "7x7 stem conv, 4 residual stages (64, 128, 256, 512 channels), 8 residual blocks, identity & 1x1 projection shortcuts."},
            {"step": 9, "title": "Rigorous Model Training", "desc": "Trained for 20 epochs with Adam optimizer, Sparse Categorical Crossentropy, ModelCheckpoint and ReduceLROnPlateau."},
            {"step": 10, "title": "Comprehensive Evaluation", "desc": "Evaluated on independent 2,700 test images: classification reports, confusion matrices, and error analysis."},
            {"step": 11, "title": "Interactive Deployment", "desc": "FastAPI inference backend and modern React dashboard for examiner demonstration and live classification."}
        ],
        "future_scope": [
            {
                "title": "Transfer Learning with Pretrained Vision Models",
                "desc": "Fine-tuning ImageNet-pretrained vision architectures (e.g. ConvNeXt, EfficientNet-V2, and Vision Transformers / ViT) to assess fine-grained transfer dynamics."
            },
            {
                "title": "Multi-Temporal Temporal Analysis",
                "desc": "Incorporating temporal image series from Sentinel-2 to monitor phenological crop changes, deforestation velocity, and urban sprawl over years."
            },
            {
                "title": "Land-Use Change Detection & Difference Mapping",
                "desc": "Developing Siamese deep networks to compute pixel-level difference maps between two satellite acquisitions of the same geographic coordinate."
            },
            {
                "title": "Explainable AI (XAI) & Heatmaps",
                "desc": "Implementing Grad-CAM (Gradient-weighted Class Activation Mapping) to highlight spectral and spatial visual regions dictating model predictions."
            },
            {
                "title": "Expansion to All 13 Sentinel-2 Spectral Bands",
                "desc": "Extending from RGB to full 13-band multispectral data (SWIR, Red Edge, NIR) for superior vegetation and water index classification (NDVI, NDWI)."
            },
            {
                "title": "Geospatial GIS Integration",
                "desc": "Connecting with OpenStreetMap / Mapbox for interactive coordinate-based satellite tile querying and real-time bounding box inference."
            }
        ]
    }

@app.get("/api/dataset-info")
def get_dataset_info():
    # Per-class distribution
    classes_data = [
        {"className": "AnnualCrop", "index": 0, "train": 2100, "val": 600, "test": 300, "total": 3000, "desc": "Cultivated agricultural land with seasonal crops (wheat, barley, corn, etc.)."},
        {"className": "Forest", "index": 1, "train": 2100, "val": 600, "test": 300, "total": 3000, "desc": "Dense natural woodland and commercial timber forests with high canopy cover."},
        {"className": "HerbaceousVegetation", "index": 2, "train": 2100, "val": 600, "test": 300, "total": 3000, "desc": "Natural grasslands, moors, meadows, and wild low-height shrublands."},
        {"className": "Highway", "index": 3, "train": 1750, "val": 500, "test": 250, "total": 2500, "desc": "High-speed multi-lane roads, motorways, highway junctions, and transport arteries."},
        {"className": "Industrial", "index": 4, "train": 1750, "val": 500, "test": 250, "total": 2500, "desc": "Manufacturing plants, storage warehouses, chemical facilities, and industrial complexes."},
        {"className": "Pasture", "index": 5, "train": 1400, "val": 400, "test": 200, "total": 2000, "desc": "Grazing land for livestock, fenced green pastures, and open grasslands."},
        {"className": "PermanentCrop", "index": 6, "train": 1750, "val": 500, "test": 250, "total": 2500, "desc": "Perennial plantations: orchards, vineyards, fruit groves, and olive trees."},
        {"className": "Residential", "index": 7, "train": 2100, "val": 600, "test": 300, "total": 3000, "desc": "Suburban and urban living areas, houses, rooftops, gardens, and residential blocks."},
        {"className": "River", "index": 8, "train": 1750, "val": 500, "test": 250, "total": 2500, "desc": "Natural running freshwater waterways, flowing rivers, canals, and riverbanks."},
        {"className": "SeaLake", "index": 9, "train": 2100, "val": 600, "test": 300, "total": 3000, "desc": "Open standing water bodies, sea coastlines, ocean bays, lakes, and reservoirs."}
    ]
    
    return {
        "dataset_name": "EuroSAT (RGB)",
        "source": "Sentinel-2 Satellite Imagery (European Space Agency / Copernicus)",
        "total_images": 27000,
        "num_classes": 10,
        "image_resolution": "64 x 64 pixels",
        "channels": 3,
        "color_space": "RGB",
        "splits": {
            "train": {"count": 18900, "percentage": 70.0},
            "val": {"count": 5400, "percentage": 20.0},
            "test": {"count": 2700, "percentage": 10.0}
        },
        "normalization": {
            "type": "Channel-wise Z-Score Standardization",
            "mean": NORM_MEAN,
            "std": NORM_STD
        },
        "augmentation": [
            "Random Horizontal Flip (tf.image.random_flip_left_right)",
            "Random Vertical Flip (tf.image.random_flip_up_down)",
            "Random Rotation up to ±18° (-0.05 to +0.05 factor)"
        ],
        "classes": classes_data
    }

@app.get("/api/models")
def get_models_info():
    return {
        "cnn": {
            "name": "Custom Baseline CNN",
            "type": "Sequential Convolutional Neural Network",
            "trained_from_scratch": True,
            "input_shape": [64, 64, 3],
            "total_parameters": 111946,
            "trainable_parameters": 111498,
            "non_trainable_parameters": 448,
            "optimizer": "Adam (initial lr=0.001)",
            "loss_function": "Sparse Categorical Crossentropy",
            "regularization": ["Batch Normalization in all conv blocks", "Dropout (rate=0.30)", "Global Average Pooling"],
            "blocks": [
                {
                    "name": "Convolutional Block 1",
                    "layers": [
                        {"type": "Conv2D", "filters": 32, "kernel": "3x3", "padding": "same", "activation": "None"},
                        {"type": "BatchNormalization", "params": 128},
                        {"type": "ReLU", "activation": "max(0, x)"},
                        {"type": "MaxPooling2D", "pool_size": "2x2", "strides": "2x2"}
                    ],
                    "output_shape": [32, 32, 32]
                },
                {
                    "name": "Convolutional Block 2",
                    "layers": [
                        {"type": "Conv2D", "filters": 64, "kernel": "3x3", "padding": "same", "activation": "None"},
                        {"type": "BatchNormalization", "params": 256},
                        {"type": "ReLU", "activation": "max(0, x)"},
                        {"type": "MaxPooling2D", "pool_size": "2x2", "strides": "2x2"}
                    ],
                    "output_shape": [16, 16, 64]
                },
                {
                    "name": "Convolutional Block 3",
                    "layers": [
                        {"type": "Conv2D", "filters": 128, "kernel": "3x3", "padding": "same", "activation": "None"},
                        {"type": "BatchNormalization", "params": 512},
                        {"type": "ReLU", "activation": "max(0, x)"},
                        {"type": "MaxPooling2D", "pool_size": "2x2", "strides": "2x2"}
                    ],
                    "output_shape": [8, 8, 128]
                },
                {
                    "name": "Classifier Head",
                    "layers": [
                        {"type": "GlobalAveragePooling2D", "desc": "Spatial reduction to 128-d feature vector"},
                        {"type": "Dense", "units": 128, "activation": "ReLU"},
                        {"type": "Dropout", "rate": 0.30, "desc": "30% regularized node deactivation"},
                        {"type": "Dense (Output)", "units": 10, "activation": "Softmax", "desc": "10-class probability distribution"}
                    ],
                    "output_shape": [10]
                }
            ]
        },
        "resnet18": {
            "name": "ResNet-18 (Deep Residual Network)",
            "type": "Residual Architecture (Implemented from Scratch)",
            "trained_from_scratch": True,
            "pretrained_weights": False,
            "input_shape": [64, 64, 3],
            "total_parameters": 11191242,
            "trainable_parameters": 11181642,
            "non_trainable_parameters": 9600,
            "optimizer": "Adam (initial lr=0.001)",
            "loss_function": "Sparse Categorical Crossentropy",
            "residual_stages": [
                {
                    "stage": "Stage 0 (Stem)",
                    "details": "Conv2D(64 filters, 7x7 kernel, stride 2) -> BatchNorm -> ReLU -> MaxPool(3x3, stride 2)",
                    "output_shape": [16, 16, 64]
                },
                {
                    "stage": "Stage 1 (Residual Blocks 1-2)",
                    "details": "2 Basic Residual Blocks with 64 filters, stride 1 (Identity shortcuts, F(x) + x)",
                    "output_shape": [16, 16, 64]
                },
                {
                    "stage": "Stage 2 (Residual Blocks 3-4)",
                    "details": "2 Basic Residual Blocks with 128 filters, stride 2 (1x1 Conv projection shortcut on block 1)",
                    "output_shape": [8, 8, 128]
                },
                {
                    "stage": "Stage 3 (Residual Blocks 5-6)",
                    "details": "2 Basic Residual Blocks with 256 filters, stride 2 (1x1 Conv projection shortcut on block 1)",
                    "output_shape": [4, 4, 256]
                },
                {
                    "stage": "Stage 4 (Residual Blocks 7-8)",
                    "details": "2 Basic Residual Blocks with 512 filters, stride 2 (1x1 Conv projection shortcut on block 1)",
                    "output_shape": [2, 2, 512]
                },
                {
                    "stage": "Classifier Head",
                    "details": "GlobalAveragePooling2D (512-d feature vector) -> Dense(10, Softmax)",
                    "output_shape": [10]
                }
            ]
        }
    }

@app.get("/api/model-metrics")
def get_model_metrics():
    # Load training histories if available
    h_cnn = {}
    h_resnet = {}
    
    cnn_hist_path = os.path.join(MODELS_DIR, "history_baseline_cnn.json")
    if os.path.exists(cnn_hist_path):
        with open(cnn_hist_path, "r") as f:
            h_cnn = json.load(f)
            
    resnet_hist_path = os.path.join(MODELS_DIR, "history_resnet18.json")
    if os.path.exists(resnet_hist_path):
        with open(resnet_hist_path, "r") as f:
            h_resnet = json.load(f)

    # Epochs data formatted for Recharts
    epochs_data = []
    num_epochs = max(len(h_cnn.get("loss", [])), len(h_resnet.get("loss", [])))
    for i in range(num_epochs):
        epochs_data.append({
            "epoch": i + 1,
            "cnn_accuracy": round(h_cnn["accuracy"][i] * 100, 2) if i < len(h_cnn.get("accuracy", [])) else None,
            "cnn_val_accuracy": round(h_cnn["val_accuracy"][i] * 100, 2) if i < len(h_cnn.get("val_accuracy", [])) else None,
            "cnn_loss": round(h_cnn["loss"][i], 4) if i < len(h_cnn.get("loss", [])) else None,
            "cnn_val_loss": round(h_cnn["val_loss"][i], 4) if i < len(h_cnn.get("val_loss", [])) else None,
            
            "resnet_accuracy": round(h_resnet["accuracy"][i] * 100, 2) if i < len(h_resnet.get("accuracy", [])) else None,
            "resnet_val_accuracy": round(h_resnet["val_accuracy"][i] * 100, 2) if i < len(h_resnet.get("val_accuracy", [])) else None,
            "resnet_loss": round(h_resnet["loss"][i], 4) if i < len(h_resnet.get("loss", [])) else None,
            "resnet_val_loss": round(h_resnet["val_loss"][i], 4) if i < len(h_resnet.get("val_loss", [])) else None,
        })

    # Per-class metrics from actual evaluation classification reports
    per_class_cnn = {
        "AnnualCrop": {"precision": 97.12, "recall": 90.00, "f1": 93.43, "support": 300},
        "Forest": {"precision": 89.52, "recall": 99.67, "f1": 94.32, "support": 300},
        "HerbaceousVegetation": {"precision": 95.08, "recall": 83.67, "f1": 89.01, "support": 300},
        "Highway": {"precision": 94.81, "recall": 80.40, "f1": 87.01, "support": 250},
        "Industrial": {"precision": 95.28, "recall": 88.80, "f1": 91.93, "support": 250},
        "Pasture": {"precision": 90.31, "recall": 88.50, "f1": 89.39, "support": 200},
        "PermanentCrop": {"precision": 85.04, "recall": 93.20, "f1": 88.93, "support": 250},
        "Residential": {"precision": 82.42, "recall": 100.00, "f1": 90.36, "support": 300},
        "River": {"precision": 86.47, "recall": 92.00, "f1": 89.15, "support": 250},
        "SeaLake": {"precision": 99.28, "recall": 92.33, "f1": 95.68, "support": 300}
    }
    
    per_class_resnet = {
        "AnnualCrop": {"precision": 95.53, "recall": 92.67, "f1": 94.08, "support": 300},
        "Forest": {"precision": 98.25, "recall": 93.67, "f1": 95.90, "support": 300},
        "HerbaceousVegetation": {"precision": 92.23, "recall": 91.00, "f1": 91.61, "support": 300},
        "Highway": {"precision": 91.60, "recall": 96.00, "f1": 93.75, "support": 250},
        "Industrial": {"precision": 95.14, "recall": 94.00, "f1": 94.57, "support": 250},
        "Pasture": {"precision": 90.95, "recall": 95.50, "f1": 93.17, "support": 200},
        "PermanentCrop": {"precision": 89.56, "recall": 89.20, "f1": 89.38, "support": 250},
        "Residential": {"precision": 93.02, "recall": 97.67, "f1": 95.28, "support": 300},
        "River": {"precision": 94.26, "recall": 92.00, "f1": 93.12, "support": 250},
        "SeaLake": {"precision": 98.33, "recall": 98.33, "f1": 98.33, "support": 300}
    }

    per_class_comparison = []
    for cls in CLASS_NAMES:
        c_m = per_class_cnn[cls]
        r_m = per_class_resnet[cls]
        per_class_comparison.append({
            "className": cls,
            "support": c_m["support"],
            "cnn_precision": c_m["precision"],
            "resnet_precision": r_m["precision"],
            "cnn_recall": c_m["recall"],
            "resnet_recall": r_m["recall"],
            "cnn_f1": c_m["f1"],
            "resnet_f1": r_m["f1"],
            "f1_improvement": round(r_m["f1"] - c_m["f1"], 2)
        })

    # Load confusion matrices
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrices.json")
    cm_data = {}
    if os.path.exists(cm_path):
        with open(cm_path, "r") as f:
            cm_data = json.load(f)

    return {
        "summary": {
            "test_samples": 2700,
            "cnn": {
                "name": "Custom Baseline CNN",
                "accuracy": 91.11,
                "loss": 0.2627,
                "precision": 91.53,
                "recall": 90.86,
                "f1_score": 90.92,
                "misclassified": 240,
                "correct": 2460,
                "error_rate": 8.89
            },
            "resnet18": {
                "name": "ResNet-18 (From Scratch)",
                "accuracy": 94.04,
                "loss": 0.1737,
                "precision": 93.89,
                "recall": 94.00,
                "f1_score": 93.92,
                "misclassified": 161,
                "correct": 2539,
                "error_rate": 5.96
            },
            "improvement": {
                "accuracy_diff": "+2.93 percentage points",
                "loss_reduction": "-0.0890 (33.9% reduction)",
                "f1_diff": "+3.00 percentage points",
                "error_reduction_count": 79,
                "error_reduction_percentage": "32.9% fewer misclassified images"
            }
        },
        "epochs_data": epochs_data,
        "per_class_comparison": per_class_comparison,
        "confusion_matrices": cm_data,
        "classes": CLASS_NAMES
    }

@app.get("/api/sample-images")
def get_sample_images(samples_per_class: int = Query(default=2, ge=1, le=5)):
    """Provides genuine sample images from each of the 10 EuroSAT classes for instant live testing."""
    samples = []
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(DATASET_DIR, cls)
        if os.path.exists(cls_dir):
            files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            files.sort()
            for f in files[:samples_per_class]:
                samples.append({
                    "className": cls,
                    "filename": f,
                    "path": f"{cls}/{f}",
                    "url": f"/static/dataset/{cls}/{f}"
                })
    return {"samples": samples, "total": len(samples)}

@app.post("/api/predict")
async def predict_image(
    file: Optional[UploadFile] = File(None),
    sample_path: Optional[str] = Form(None),
    model_name: str = Form("resnet18")
):
    """
    Runs actual neural network inference on an uploaded satellite image or selected EuroSAT sample.
    Returns predicted class, confidence, latency, and full probability distribution over all 10 classes.
    """
    t_start = time.time()
    
    # 1. Validate model selection
    model_key = model_name.lower().replace("-", "").strip()
    if model_key in ["resnet", "resnet18"]:
        selected_key = "resnet18"
        display_model_name = "ResNet-18 (From Scratch)"
    elif model_key in ["cnn", "baseline", "baselinecnn"]:
        selected_key = "cnn"
        display_model_name = "Custom Baseline CNN"
    else:
        raise HTTPException(status_code=400, detail=f"Invalid model '{model_name}'. Choose 'resnet18' or 'cnn'.")

    model = loaded_models.get(selected_key)
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model '{display_model_name}' is not currently loaded.")

    # 2. Acquire Image bytes
    img = None
    source_type = "upload"
    image_name = "uploaded_image.jpg"
    
    if file is not None and file.filename:
        # Check file extension
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]:
            raise HTTPException(status_code=400, detail=f"Unsupported image format '{ext}'. Upload JPG, PNG, or TIFF.")
        contents = await file.read()
        try:
            img = Image.open(io.BytesIO(contents))
            image_name = file.filename
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Corrupt or unreadable image: {e}")
    elif sample_path:
        # Load from dataset
        safe_rel_path = os.path.normpath(sample_path).replace("\\", "/").lstrip("/")
        full_sample_path = os.path.join(DATASET_DIR, safe_rel_path)
        if not os.path.exists(full_sample_path):
            raise HTTPException(status_code=404, detail=f"Sample image '{safe_rel_path}' not found on server.")
        try:
            img = Image.open(full_sample_path)
            source_type = "dataset_sample"
            image_name = os.path.basename(safe_rel_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load sample image: {e}")
    else:
        raise HTTPException(status_code=400, detail="No image provided. Please upload an image or select a dataset sample.")

    # 3. Preprocess
    orig_width, orig_height = img.size
    tensor_input = preprocess_pil_image(img)

    # 4. Run Model Prediction
    t_inf_start = time.time()
    raw_probs = model.predict(tensor_input, verbose=0)[0]
    inference_time_ms = round((time.time() - t_inf_start) * 1000, 2)
    total_time_ms = round((time.time() - t_start) * 1000, 2)

    # 5. Extract probabilities and ranked predictions
    pred_idx = int(np.argmax(raw_probs))
    confidence = float(raw_probs[pred_idx])
    predicted_class = CLASS_NAMES[pred_idx]

    probabilities = []
    for idx, cls in enumerate(CLASS_NAMES):
        prob = float(raw_probs[idx])
        probabilities.append({
            "className": cls,
            "index": idx,
            "probability": round(prob, 6),
            "percentage": round(prob * 100, 2)
        })

    # Sort probabilities in descending order
    ranked_probabilities = sorted(probabilities, key=lambda x: x["probability"], reverse=True)

    return {
        "status": "success",
        "predicted_class": predicted_class,
        "class_index": pred_idx,
        "confidence": round(confidence, 6),
        "confidence_percentage": round(confidence * 100, 2),
        "model_used": {
            "key": selected_key,
            "name": display_model_name,
            "parameters": 11191242 if selected_key == "resnet18" else 111946
        },
        "image_metadata": {
            "name": image_name,
            "source": source_type,
            "original_dimensions": f"{orig_width} x {orig_height}",
            "processed_dimensions": "64 x 64 x 3"
        },
        "latency": {
            "inference_time_ms": inference_time_ms,
            "total_time_ms": total_time_ms
        },
        "probabilities": probabilities,
        "ranked_probabilities": ranked_probabilities
    }

# Mount built React frontend if dist directory exists
FRONTEND_DIST = os.path.join(BASE_DIR, "frontend", "dist")
if os.path.exists(FRONTEND_DIST):
    assets_dir = os.path.join(FRONTEND_DIST, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")
    
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path.startswith("api") or full_path.startswith("static"):
            raise HTTPException(status_code=404, detail="API route not found")
        file_path = os.path.join(FRONTEND_DIST, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8000, reload=False)
