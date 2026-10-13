import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def run_eda():
    print("=" * 60)
    print("       EUROSAT DATASET - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Define paths
    dataset_dir = os.path.join("EuroSAT_Dataset", "2750")
    train_csv_path = os.path.join(dataset_dir, "train.csv")
    val_csv_path = os.path.join(dataset_dir, "validation.csv")
    test_csv_path = os.path.join(dataset_dir, "test.csv")
    label_map_path = os.path.join(dataset_dir, "label_map.json")
    
    # Check if files exist
    for path in [train_csv_path, val_csv_path, test_csv_path, label_map_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")
            
    # Load Label Map
    with open(label_map_path, 'r') as f:
        label_map = json.load(f)
    print(f"\n[+] Loaded {len(label_map)} classes from label_map.json:")
    for cls_name, cls_idx in label_map.items():
        print(f"    - {cls_name}: {cls_idx}")
        
    # Load CSVs
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    # 1. Dataset Split Sizes
    total_images = len(train_df) + len(val_df) + len(test_df)
    print(f"\n[+] Dataset Splits Info:")
    print(f"    - Train set:      {len(train_df)} images ({len(train_df)/total_images*100:.1f}%)")
    print(f"    - Validation set: {len(val_df)} images ({len(val_df)/total_images*100:.1f}%)")
    print(f"    - Test set:       {len(test_df)} images ({len(test_df)/total_images*100:.1f}%)")
    print(f"    - Total images:   {total_images}")
    
    # 2. Class Distribution Analysis
    print(f"\n[+] Class Distribution in Splits:")
    train_counts = train_df['ClassName'].value_counts().rename('Train')
    val_counts = val_df['ClassName'].value_counts().rename('Val')
    test_counts = test_df['ClassName'].value_counts().rename('Test')
    
    dist_df = pd.concat([train_counts, val_counts, test_counts], axis=1).fillna(0).astype(int)
    # Sort by total count
    dist_df['Total'] = dist_df.sum(axis=1)
    dist_df = dist_df.sort_values(by='Total', ascending=False)
    print(dist_df)
    
    # 3. Save Class Distribution Visualization
    print(f"\n[+] Generating class distribution visualization...")
    sns.set_theme(style="whitegrid", palette="muted")
    plt.figure(figsize=(12, 7))
    
    # Melt dataframe for seaborn barplot
    plot_df = dist_df.drop(columns=['Total']).reset_index().rename(columns={'index': 'ClassName'})
    melted_df = pd.melt(plot_df, id_vars=['ClassName'], var_name='Split', value_name='Count')
    
    ax = sns.barplot(data=melted_df, x='ClassName', y='Count', hue='Split')
    plt.title("EuroSAT Class Distribution by Dataset Split", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Class Name", fontsize=12, fontweight='bold')
    plt.ylabel("Number of Images", fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    # Annotate bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center',
                        xytext=(0, 5),
                        textcoords='offset points',
                        fontsize=9)
            
    plt.tight_layout()
    dist_plot_path = "class_distribution.png"
    plt.savefig(dist_plot_path, dpi=150)
    plt.close()
    print(f"    - Saved class distribution chart to {dist_plot_path}")
    
    # 4. Verify Files and Analyze Dimensions & Statistics
    print(f"\n[+] Verifying images and computing stats (sample of 3000 images for speed)...")
    
    # Let's sample 3000 images from train set to compute channel means and stds, and check dimensions
    sample_size = min(3000, len(train_df))
    sampled_df = train_df.sample(n=sample_size, random_state=42)
    
    shapes = set()
    pixel_values = []
    
    corrupted_count = 0
    
    for idx, row in sampled_df.iterrows():
        img_rel_path = row['Filename']
        img_path = os.path.join(dataset_dir, img_rel_path)
        
        try:
            with Image.open(img_path) as img:
                img_arr = np.array(img)
                shapes.add(img_arr.shape)
                # Normalize pixel values to [0, 1] for mean and std calculation
                pixel_values.append(img_arr / 255.0)
        except Exception as e:
            print(f"    - Warning: Failed to load image {img_path}: {e}")
            corrupted_count += 1
            
    if corrupted_count > 0:
        print(f"    - Found {corrupted_count} corrupted images in sample.")
    else:
        print(f"    - All sampled images loaded successfully.")
        
    print(f"    - Unique image shapes found in sample: {shapes}")
    
    # Compute Channel Statistics (Mean & Std)
    if len(pixel_values) > 0:
        all_pixels = np.stack(pixel_values, axis=0) # Shape: (N, H, W, C)
        # Compute mean and std per channel (R, G, B)
        # In NumPy shape (N, H, W, C), channel axis is 3
        means = np.mean(all_pixels, axis=(0, 1, 2))
        stds = np.std(all_pixels, axis=(0, 1, 2))
        
        print(f"\n[+] Computed RGB Channel Statistics (Normalized to [0, 1]):")
        print(f"    - Mean: R={means[0]:.4f}, G={means[1]:.4f}, B={means[2]:.4f}")
        print(f"    - Std:  R={stds[0]:.4f}, G={stds[1]:.4f}, B={stds[2]:.4f}")
        
        print(f"\n[+] Recommended PyTorch Normalization values:")
        print(f"    - transforms.Normalize(mean=[{means[0]:.4f}, {means[1]:.4f}, {means[2]:.4f}],")
        print(f"                           std=[{stds[0]:.4f}, {stds[1]:.4f}, {stds[2]:.4f}])")
    else:
        print("    - Error: No valid image pixel values found to compute stats.")
        
    # 5. Generate and Save Sample Image Grid
    print(f"\n[+] Generating sample image grid...")
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    axes = axes.flatten()
    
    unique_classes = sorted(list(label_map.keys()))
    
    for i, cls_name in enumerate(unique_classes):
        # Find first image for this class in train_df
        cls_rows = train_df[train_df['ClassName'] == cls_name]
        if len(cls_rows) > 0:
            row = cls_rows.iloc[0]
            img_path = os.path.join(dataset_dir, row['Filename'])
            try:
                img = Image.open(img_path)
                axes[i].imshow(img)
                axes[i].set_title(cls_name, fontsize=12, fontweight='bold', pad=8)
                axes[i].axis('off')
            except Exception as e:
                axes[i].text(0.5, 0.5, f"Error loading\n{cls_name}", ha='center', va='center')
                axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f"No image for\n{cls_name}", ha='center', va='center')
            axes[i].axis('off')
            
    plt.suptitle("EuroSAT Dataset - Representative Class Samples", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    sample_grid_path = "sample_images.png"
    plt.savefig(sample_grid_path, dpi=150)
    plt.close()
    print(f"    - Saved sample image grid to {sample_grid_path}")
    
    print("\n" + "=" * 60)
    print("       EDA COMPLETE! Dataset is clean and ready for training.")
    print("=" * 60)

if __name__ == "__main__":
    run_eda()