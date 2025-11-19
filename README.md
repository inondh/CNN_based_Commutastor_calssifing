# CNN-Based Commutator Classifier

A deep learning tool that analyzes mathematical interactions between 2D Gaussian signals by computing their commutator (XY - YX) and using a CNN to classify signal similarity.

## What It Does

This project:
1. Generates ~1,000 pairs of 2D Gaussian signals with varying positions and widths
2. Computes the commutator (XY - YX) for each pair to measure non-commutativity
3. Trains a CNN to automatically classify whether signal pairs are "similar" or "dissimilar"
4. Exports comprehensive results including visualizations and Excel spreadsheets

## Installation

```bash
pip install numpy matplotlib tensorflow keras scikit-learn opencv-python seaborn pandas openpyxl pillow
```

## Quick Start

1. **Set output path** in `main_analysis.py` (line 580):
```python
output_dir = Path(r"C:/Your/Output/Path")
```

2. **Run the analysis**:
```bash
python CNN commutator based simulation.py
```

3. **Check results** in the timestamped output folder

## Key Outputs

- **CNN_Results_Summary.xlsx** - Complete results with CNN predictions and overlap metrics
- **Commutator visualizations** - PNG files showing signal pairs and their commutators
- **learning_curves.png** - Model training performance
- **best_model.h5** - Trained CNN model


# TIFF Image Preprocessing Tool

## What It Does

This script processes 48 TIFF  images by:
- Cropping to a specific region of interest (ROI)
- Normalizing intensity using percentile-based contrast enhancement
- Saving with standardized filenames and compression

## Installation

```bash
pip install pillow numpy
```

## Quick Start

1. **Set your output path** (line 9):
```python
output_dir = r"C:/Your/Output/Path"
```

2. **Ensure DATA folder structure exists**:
```
./DATA/
├── g1/, g2/, g3/, g4/, g5/, g6/  # 'g' sample groups
└── b1/, b2/, b3/, b4/, b5/, b6/  # 'b' sample groups
```

3. **Run the script**:
```bash
python Tiff image initial processing .py
```

## Configuration

### Cropping Parameters
```python
Y_CENTER = 1150           # Vertical center point
X_RANGE = [560, 1260]     # Horizontal: 700 pixels wide
Y_RANGE = [950, 1350]     # Vertical: 400 pixels tall
```



## Input Files

- **48 TIFF images** from 12 samples (g1-g6, b1-b6)
- **Z-depths**: z54, z55, z56, z57 (different focal planes)
- **Format**: `.tif` files with naming pattern: `{sample}_z{depth}_th08_s400_part1.tif`

## Output

### Processed Images
- **Filename format**: `RDA_{color}{number}_z{depth}.tiff`
- **Examples**: `RDA_g1_z54.tiff`, `RDA_b3_z56.tiff`


### Console Output
```
✓ Saved: RDA_g1_z54.tiff
✓ Saved: RDA_g1_z55.tiff
...
--- Summary ---
Successfully processed: 48
Errors: 0
Output directory: C:/Your/Output/Path
```
# Commutator Analysis with Data Augmentation

A tool for analyzing  images by computing matrix commutators (XY - YX) and generating augmented training datasets with enhanced blur variations.

## What It Does

This script:
1. Loads "good" and "bad" scans from preprocessed TIFF files
2. Creates a reference matrix (X) by averaging randomly selected good scans
3. Computes commutators with remaining scans to identify differences
4. Generates 20 augmented versions of each commutator for ML training
5. Applies 5 types of blur augmentation (light/medium/heavy Gaussian, motion, defocus)

## Installation

```bash
pip install numpy pillow matplotlib scipy
```

## Quick Start

1. **Set your directories** (lines 10-12):
```python
input_dir = r"C:/Path/To/Cropped/Images"  # From Tiff image initial processing .py output
base_output_dir = r"C:/Path/To/Output"
```

2. **Run the script**:
```bash
python Commutators.py
```

3. **Check outputs** in the timestamped output folder

## Configuration



### Preprocessing Settings
```python
CROP_GRAY_AREAS = True          # Remove padding from commutators
GRAY_THRESHOLD = 10             # Pixel threshold for cropping
```

## Input Files

Expects preprocessed TIFF images with naming format: `RDA_{g|b}{number}_z{depth}.tiff`

**Example structure:**
```
input_dir/
├── RDA_g1_z54.tiff, RDA_g1_z55.tiff, ...  # Good scans
└── RDA_b1_z54.tiff, RDA_b1_z55.tiff, ...  # Bad scans
```

## Output Structure

```
output_dir/
├── X_averaged_random_selection.tiff        # Reference matrix
├── Comm_vs_g2_z55_GOOD.tiff               # Original commutators (good)
├── Comm_vs_b3_z56_BAD.tiff                # Original commutators (bad)
├── selected_scans.txt                      # List of scans used for X
├── ANALYSIS_SUMMARY.txt                    # Complete analysis report
│
├── Augmented/                              # Augmented dataset
│   ├── Comm_vs_g2_z55_GOOD_aug1_hflip_blur_light.tiff
│   ├── Comm_vs_g2_z55_GOOD_aug2_rot90_sp.tiff
│   └── ... (~800 augmented files)
│
└── X_Commutator_Pairs/                     # Visualization images
    ├── Pair_Comm_vs_g2_z55_GOOD.png
    └── ... (shows X, Y, and commutator side-by-side)
```

## Key Features

### Commutator Computation
- **Matrix operation**: XY - YX reveals non-commutativity
- **Zero-padding**: Automatically handles non-square matrices
- **Cropping**: Removes gray padding areas from results

### Enhanced Blur Augmentation
Five blur types simulate real-world variations:
- **Light Gaussian** (σ=0.3-0.5): Subtle defocus
- **Medium Gaussian** (σ=0.6-0.8): Moderate defocus  
- **Heavy Gaussian** (σ=0.9-1.2): Strong defocus
- **Motion blur**: Simulates camera/sample movement
- **Defocus blur**: Circular out-of-focus effect

### Other Augmentations
- Horizontal/vertical flips
- 90°/180°/270° rotations
- Salt & pepper noise
- All augmentations applied probabilistically

### Reproducibility
- Uses fixed random seed (42) for consistent scan selection
- Detailed logging of selected scans and augmentation statistics

## Expected Dataset Size

With default settings:
- **Original commutators**: 40 (from good + bad scans)
- **Augmented versions**: 800 (20 per original)
- **Total dataset**: 840 images (21× multiplier)


## Understanding the Output

### Commutator Interpretation
- **Small values**: X and Y are similar (nearly commute)
- **Large values**: X and Y are different (strong non-commutativity)

### Filename Convention
```
Comm_vs_g2_z55_GOOD_aug5_hflip_rot90_blur_med.tiff
         │  │   │    │    │     │     └─ Blur type
         │  │   │    │    │     └─ 90° rotation
         │  │   │    │    └─ Horizontal flip
         │  │   │    └─ Augmentation number
         │  │   └─ Label (GOOD/BAD)
         │  └─ Z-depth
         └─ Chip number
```

## Blur Statistics

The script tracks and reports:
- Count of each blur type applied
- Percentage of images with blur
- Distribution across augmentation types
## Analysis Summary

The script generates `ANALYSIS_SUMMARY.txt` with:
- Complete augmentation settings
- Selected scans used for averaging
- Dataset statistics (original vs augmented counts)
- Blur type distribution
- Output file structure

# CNN Classifier for Commutator 

A deep learning pipeline that trains a Convolutional Neural Network to classify microscopy scan quality (GOOD vs BAD) using commutator patterns.

## What It Does

This script:
1. Loads original and augmented commutator images from previous analysis
2. Builds a 4-block deep CNN (64→128→256→512 filters)
3. Trains with batch size 16 for balanced performance
4. Generates comprehensive evaluation metrics and visualizations
5. Exports detailed predictions to Excel with per-sample analysis

## Installation

```bash
pip install numpy pillow matplotlib seaborn tensorflow keras scikit-learn pandas openpyxl
```

**GPU Support** (recommended for faster training):
```bash
pip install tensorflow-gpu
```

## Quick Start

1. **Set your directories** (lines 20-21):
```python
commutator_dir = r"C:/Path/To/Commutator/Output"  # From commutator_analysis.py
# augmented_dir automatically set to: commutator_dir/Augmented
```

2. **Run training**:
```bash
python CNN classifier.py
```

3. **Check results** in timestamped output folder (e.g., `CNN_Results/20231119_143052/`)

## Key Optimizations


### Model Architecture
```
Deep CNN with 4 Convolutional Blocks:
├── Block 1: Conv2D(64) → BatchNorm → ReLU → MaxPool → Dropout(0.1)
├── Block 2: Conv2D(128) → BatchNorm → ReLU → MaxPool → Dropout(0.1)
├── Block 3: Conv2D(256) → BatchNorm → ReLU → MaxPool → Dropout(0.15)
├── Block 4: Conv2D(512) → BatchNorm → ReLU → GlobalAvgPool → Dropout(0.2)
└── Dense(256) → Dense(128) → Dense(1, sigmoid)
```


## Expected Dataset

### Input Requirements
- **Original commutators**: 40 from commutator_analysis.py
- **Augmented versions**: 800 (20× multiplier with blur variations)
- **Total expected**: 840 images minimum

### Image Specifications
- **Format**: TIFF files
- **Naming**: `Comm_vs_{chip}_GOOD.tiff` or `Comm_vs_{chip}_BAD.tiff`
- **Processing**: Auto-resized to 128×128, normalized to [0,1]

## Output Files

### Timestamped Output Directory
```
CNN_Results/20231119_143052/
├── best_model.h5                    # Best model (highest val_auc)
├── final_model.h5                   # Final trained model
├── training_log.csv                 # Epoch-by-epoch metrics
│
├── learning_curves.png              # Loss, accuracy, precision, recall, AUC
├── confusion_matrix.png             # True vs predicted labels
├── roc_curve.png                    # ROC curve with AUC score
├── results_distribution.png         # 4-panel CNN output analysis
│
├── detailed_predictions.xlsx        # ⭐ Main results file
│   ├── All_Predictions              # Every test sample
│   ├── Correct                      # Correctly classified
│   ├── Misclassified                # Errors for analysis
│   └── Summary                      # Complete statistics
│
└── classification_report.txt        # Precision/recall/F1 per class
```

## Excel Output Columns

**detailed_predictions.xlsx** includes:
- `Filename`: Original file name
- `Chip_ID`: Chip identifier
- `True_Type`: Actual label (GOOD/BAD)
- `CNN_Raw_Output`: Model confidence (0-1)
- `CNN_Prediction`: Binary prediction
- `Correct`: True/False classification
- `Confidence`: Prediction confidence level
- `Source`: Original or augmented

## Training Parameters



## Performance Metrics

The model reports:
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

### Interpretation
```
CNN Output > 0.5 → Chip is GOOD
CNN Output < 0.5 → Chip is BAD
```

## Results Interpretation

### Excellent Performance
- **Good separation**: Good chips average >0.7, Bad chips <0.3
- **High AUC**: >0.95
- **Class-wise accuracy**: Both >85%

### Acceptable Performance
- **Moderate separation**: Difference >0.3 between classes
- **AUC**: 0.85-0.95
- **Balanced accuracy**: Both classes >75%

### Needs Improvement
- **Poor separation**: Difference <0.15
- **AUC**: <0.85
- **Imbalanced**: One class much worse than other

## Visualizations

### 1. Learning Curves
- Training vs validation for all metrics
- Shows overfitting or underfitting trends
- Helps tune hyperparameters

### 2. Confusion Matrix
- True positives/negatives
- False positives/negatives
- Class-wise performance




## Memory Optimization

The script includes GPU memory management:
```python
# Automatic GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
```

If memory issues occur:
- Reduce `IMG_SIZE` from 128 to 96 or 64
- Decrease `BATCH_SIZE` from 16 to 8
- Use CPU-only: `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`
