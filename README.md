# DeepSat: Land Use and Land Cover (LULC) Satellite Vision

DeepSat is a deep learning computer vision pipeline designed for Land Use and Land Cover (LULC) classification using satellite imagery from the **EuroSAT** dataset (Sentinel-2 satellite images). The repository implements custom baseline Convolutional Neural Networks (CNN) and ResNet-18 architectures, complete with data loaders, exploratory data analysis (EDA), data augmentation, training routines, and comprehensive evaluation metrics.

---

## 📊 Dataset: EuroSAT

The [EuroSAT](https://github.com/phelber/eurosat) dataset consists of Sentinel-2 satellite images covering 13 spectral bands and RGB images across 10 Land Use and Land Cover classes:
- Annual Crop
- Forest
- Herbaceous Vegetation
- Highway
- Industrial
- Pasture
- Permanent Crop
- Residential
- River
- Sea Lake

Image Resolution: `64 x 64` pixels.

---

## 🚀 Model Benchmarks

Both models were trained and benchmarked on an identical test split:

| Model | Test Accuracy | Test Loss | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Custom Baseline CNN** | 91.11% | 0.2627 | 91.53% | 90.86% | 90.92% |
| **ResNet-18** | **94.04%** | **0.1737** | **93.89%** | **94.00%** | **93.92%** |

---

## 📁 Repository Structure

```text
├── baseline_cnn.py          # Baseline CNN architecture definition
├── resnet18.py              # ResNet-18 model architecture
├── data_loader.py           # tf.data pipeline with augmentation & prefetching
├── download_data.py         # Automated EuroSAT dataset download script
├── eda_and_preprocessing.py # Exploratory Data Analysis & visual inspections
├── train_baseline.py        # Training script for Baseline CNN
├── train_resnet18.py        # Training script for ResNet-18
├── evaluate_baseline.py     # Evaluation routine for Baseline CNN
├── evaluate_models.py       # Comparative evaluation and classification reports
├── predict.py               # Single image / batch inference script
├── test_model.py            # Quick model test verification
├── show_baseline_model.py   # Baseline model summary & visual architecture graph
├── requirements.txt         # Required Python dependencies
├── models/                  # Training history and metadata (weights gitignored)
├── plots/                   # Loss curves, accuracy plots, confusion matrices
└── results/                 # Metrics, classification reports, CSV results
```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Farhan-Ahmed-code/Deepsat.git
   cd Deepsat
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 Usage

### 1. Download Dataset
Download and extract the EuroSAT dataset:
```bash
python download_data.py
```

### 2. Exploratory Data Analysis & Preprocessing
Inspect class distribution, image dimensions, and generate sample visualizations:
```bash
python eda_and_preprocessing.py
```

### 3. Model Training
Train the Baseline CNN:
```bash
python train_baseline.py
```

Train the ResNet-18 model:
```bash
python train_resnet18.py
```

### 4. Evaluation & Comparison
Evaluate both models on the held-out test set and produce comparison tables, confusion matrices, and classification reports:
```bash
python evaluate_models.py
```

### 5. Inference / Prediction
Run inference on any satellite image:
```bash
python predict.py --image_path path/to/sample.jpg --model_type resnet18
```

---

## 📈 Visualizations & Artifacts

- **Training Curves**: Generated in `plots/` (`accuracy_*.png`, `loss_*.png`).
- **Confusion Matrices**: Visualized in `plots/` (`confusion_matrix_*.png`).
- **Classification Reports & Metrics**: Stored in `results/`.
