import os
import sys

# Suppress TensorFlow C++ engine and python-level logs/warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Define paths
MODEL_PATH = os.path.join("models", "best_baseline_cnn.keras")
PLOT_PATH = "baseline_cnn_architecture.png"

def print_separator(char="=", length=75):
    print(char * length)

def load_and_summarize_model():
    print_separator()
    print("                EUROSAT BASELINE CNN MODEL VIEWER & DOCUMENTATION")
    print_separator()
    
    # 1. Verify and Load the Model
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Trained model file not found at: {MODEL_PATH}")
        print("Please ensure the model is trained and saved in the models directory first.")
        sys.exit(1)
        
    print(f"[+] Successfully located trained model at: {MODEL_PATH}")
    print("[+] Loading Keras model checkpoint...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("[+] Model loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to load the model checkpoint: {e}")
        sys.exit(1)
        
    # 2. Print Programmatic Layer Table
    print("\n" + "=" * 35 + " LAYER DETAILS " + "=" * 35)
    print(f"{'Layer Name':<28} | {'Layer Type':<25} | {'Output Shape':<22} | {'Params':<10}")
    print("-" * 96)
    
    for layer in model.layers:
        name = layer.name
        l_type = type(layer).__name__
        try:
            shape = str(layer.output.shape)
        except Exception:
            shape = 'Unknown'
        params = layer.count_params()
        print(f"{name:<28} | {l_type:<25} | {shape:<22} | {params:<10,}")
        
    print("-" * 96)
    
    # Extract Parameter Counts
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    
    print(f"Total Parameters:          {total_params:,}")
    print(f"Trainable Parameters:      {trainable_params:,}")
    print(f"Non-Trainable Parameters:  {non_trainable_params:,}")
    print("=" * 96)
    
    # 3. Print Keras Built-in Summary
    print("\n[+] Displaying official Keras model.summary():")
    model.summary()
    
    # 4. Generate Visual Architecture Diagram
    generate_visualization(model)
    
    # 5. Print Architecture Explanation
    print_architecture_explanation()

def generate_visualization(model):
    print("\n" + "=" * 35 + " ARCHITECTURE DIAGRAM " + "=" * 35)
    print("[+] Attempting to generate model architecture visualization...")
    
    # Check if pydot / graphviz are available
    pydot_available = True
    try:
        import pydot
    except ImportError:
        pydot_available = False
        
    if not pydot_available:
        print("[!] NOTICE: Python package 'pydot' is not installed.")
        print("    In accordance with project rules, the environment was NOT modified to install it.")
        print("    Falling back to custom matplotlib-based architecture generator...")
    else:
        # Try to run keras.utils.plot_model as verification
        try:
            tf.keras.utils.plot_model(model, to_file="keras_plot_temp.png", show_shapes=True, show_layer_names=True)
            # Remove temp file if created
            if os.path.exists("keras_plot_temp.png"):
                os.remove("keras_plot_temp.png")
            print("[+] Keras plot_model check passed successfully.")
        except Exception as e:
            print(f"[!] NOTICE: Keras plot_model failed to execute: {e}")
            print("    This is typically due to missing Graphviz system binaries on your PATH.")
            print("    In accordance with project rules, the environment was NOT modified.")
            print("    Falling back to custom matplotlib-based architecture generator...")
            
    # Generate high-quality matplotlib diagram
    try:
        draw_matplotlib_diagram(model)
    except Exception as e:
        print(f"[ERROR] Failed to draw matplotlib diagram: {e}")

def draw_matplotlib_diagram(model):
    # Setup layout
    layers_info = []
    
    # Input layer first
    input_shape = model.input_shape
    layers_info.append({
        'name': 'input_layer',
        'type': 'Input',
        'shape': str(input_shape),
        'params': 0
    })
    
    for layer in model.layers:
        name = layer.name
        l_type = type(layer).__name__
        try:
            shape = str(layer.output.shape)
        except Exception:
            shape = 'Unknown'
        params = layer.count_params()
        layers_info.append({
            'name': name,
            'type': l_type,
            'shape': shape,
            'params': params
        })
        
    num_layers = len(layers_info)
    
    # Set height dynamically based on layers
    fig_height = (num_layers + 1) * 1.3 + 1.0
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis('off')
    
    # Beautiful color palette for design aesthetics
    colors = {
        'Input': '#475569',                  # Dark Slate
        'Conv2D': '#3b82f6',                 # Royal Blue
        'BatchNormalization': '#6b7280',     # Gray
        'ReLU': '#10b981',                   # Emerald Green
        'MaxPooling2D': '#f97316',           # Orange
        'GlobalAveragePooling2D': '#8b5cf6', # Violet
        'Dense': '#eab308',                  # Yellow
        'Dropout': '#ef4444',                # Red
    }
    
    text_colors = {
        'Input': '#ffffff',
        'Conv2D': '#ffffff',
        'BatchNormalization': '#ffffff',
        'ReLU': '#ffffff',
        'MaxPooling2D': '#ffffff',
        'GlobalAveragePooling2D': '#ffffff',
        'Dense': '#000000',
        'Dropout': '#ffffff',
    }
    
    y_max = (num_layers + 1) * 1.5 + 0.5
    ax.set_ylim(0, y_max)
    ax.set_xlim(0, 10)
    
    # Custom Title
    plt.title("EuroSAT Baseline CNN Model Architecture", fontsize=18, fontweight='bold', pad=20, color='#1e293b')
    
    box_width = 7.5
    box_height = 0.95
    center_x = 5.0
    
    # Plot layers
    for i, info in enumerate(layers_info):
        y_pos = y_max - (i * 1.5) - 1.2
        
        l_type = info['type']
        bg_color = colors.get(l_type, '#a8a29e')
        t_color = text_colors.get(l_type, '#000000')
        
        # Fancy Bounding Box
        rect = patches.FancyBboxPatch(
            (center_x - box_width/2, y_pos - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.1",
            linewidth=1.5,
            edgecolor='#1e293b',
            facecolor=bg_color,
            mutation_scale=0.4
        )
        ax.add_patch(rect)
        
        # Labels inside boxes
        name_str = f"{info['name']} ({l_type})" if l_type != 'Input' else "Input Stage"
        shape_str = f"Output Shape: {info['shape']}"
        param_str = f"Params: {info['params']:,}" if info['params'] > 0 else "Params: 0"
        
        ax.text(center_x, y_pos + 0.20, name_str, ha='center', va='center', fontsize=11, fontweight='bold', color=t_color)
        ax.text(center_x, y_pos - 0.05, shape_str, ha='center', va='center', fontsize=9.5, color=t_color)
        ax.text(center_x, y_pos - 0.28, param_str, ha='center', va='center', fontsize=9, style='italic', color=t_color)
        
        # Draw Arrow
        if i < num_layers - 1:
            next_y_pos = y_max - ((i + 1) * 1.5) - 1.2
            arrow_start_y = y_pos - box_height/2 - 0.05
            arrow_end_y = next_y_pos + box_height/2 + 0.05
            
            ax.annotate('', xy=(center_x, arrow_end_y), xytext=(center_x, arrow_start_y),
                        arrowprops=dict(arrowstyle="->", color='#475569', lw=2, mutation_scale=15))
            
    # Draw Model Parameter Summary Card
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    
    y_pos = y_max - (num_layers * 1.5) - 1.2
    
    rect_summary = patches.FancyBboxPatch(
        (center_x - box_width/2, y_pos - box_height/2),
        box_width, box_height,
        boxstyle="round,pad=0.1",
        linewidth=2.0,
        edgecolor='#475569',
        facecolor='#f8fafc',
        mutation_scale=0.4
    )
    ax.add_patch(rect_summary)
    
    ax.text(center_x, y_pos + 0.22, "Model Parameters Summary", ha='center', va='center', fontsize=12, fontweight='bold', color='#1e293b')
    ax.text(center_x, y_pos - 0.05, f"Total: {total_params:,} parameters", ha='center', va='center', fontsize=10, fontweight='bold', color='#475569')
    ax.text(center_x, y_pos - 0.28, f"Trainable: {trainable_params:,}  |  Non-Trainable: {non_trainable_params:,}", ha='center', va='center', fontsize=9, color='#64748b')
    
    # Arrow to summary card
    last_layer_y_pos = y_max - ((num_layers - 1) * 1.5) - 1.2
    arrow_start_y = last_layer_y_pos - box_height/2 - 0.05
    arrow_end_y = y_pos + box_height/2 + 0.05
    ax.annotate('', xy=(center_x, arrow_end_y), xytext=(center_x, arrow_start_y),
                arrowprops=dict(arrowstyle="->", color='#475569', lw=2, mutation_scale=15))

    # Save to file
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Architecture diagram saved successfully as: {PLOT_PATH}")
    print("=" * 96)

def print_architecture_explanation():
    print("\n" + "=" * 35 + " ARCHITECTURE EXPLANATION " + "=" * 35)
    explanation = """
1. What the Input Is:
   - The input is RGB satellite images from the EuroSAT dataset.
   - The shape is (64, 64, 3), indicating 64x64 pixel spatial dimensions with 3 color channels (Red, Green, Blue).

2. What the Convolution Blocks Do:
   - The model contains 3 stacked Convolution blocks (Conv2D -> BatchNormalization -> ReLU -> MaxPooling2D).
   - Conv2D layers apply 3x3 filters (32, 64, and 128 filters respectively) to extract local spatial features:
     * Lower layers (Conv1) extract basic edges, shapes, and textures.
     * Deeper layers (Conv2 & Conv3) assemble these into higher-level features representing structures, fields, roads, or land patterns.
   - BatchNormalization scales intermediate layer outputs, stabilizing and accelerating training.
   - ReLU (Rectified Linear Unit) introduces non-linearity, enabling the model to learn complex, non-linear relationships.

3. What Pooling Does:
   - MaxPooling2D layers with a pool size of 2x2 reduce the spatial resolution of the feature maps by half:
     * (64, 64) -> (32, 32) -> (16, 16) -> (8, 8).
   - This progressive downsampling keeps only the most prominent feature responses, introduces spatial translation invariance, 
     and reduces computational complexity and memory footprint.
   - GlobalAveragePooling2D is used at the transition to the classification head:
     * It averages each of the 128 feature maps of size (8, 8) into a single scalar value.
     * This compresses spatial features into a 128-dimensional vector, replacing the traditional flatten layer and 
       drastically reducing the number of dense connections, which helps prevent overfitting.

4. What the Final Layer Does:
   - The head consists of a fully connected Dense layer (128 units, ReLU), a Dropout layer (30% dropout rate for regularisation),
     and a final output Dense layer.
   - The final Dense layer outputs raw logits which are passed through the Softmax activation function.
   - The Softmax function normalises the outputs into a probability distribution summing to 1.0, representing the model's 
     confidence for each of the target categories.

5. Why There Are 10 Output Classes:
   - The model has 10 output classes because the EuroSAT land cover dataset categorises remote sensing imagery into exactly 
     10 distinct LULC (Land Use / Land Cover) categories:
     1. Annual Crop
     2. Forest
     3. Herbaceous Vegetation
     4. Highway
     5. Industrial
     6. Pasture
     7. Permanent Crop
     8. Residential
     9. River
     10. Sea Lake
"""
    print(explanation.strip())
    print("=" * 96)

if __name__ == "__main__":
    load_and_summarize_model()
