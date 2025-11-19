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

1. **Set output path** in `main_analysis.py` (line ~580):
```python
output_dir = Path(r"C:/Your/Output/Path")
```

2. **Run the analysis**:
```bash
python main_analysis.py
```

3. **Check results** in the timestamped output folder

## Key Outputs

- **CNN_Results_Summary.xlsx** - Complete results with CNN predictions and overlap metrics
- **Commutator visualizations** - PNG files showing signal pairs and their commutators
- **learning_curves.png** - Model training performance
- **best_model.h5** - Trained CNN model

## Applications

- Quantum signal processing
- Phase space analysis
- Interference pattern recognition
- Time-frequency analysis (Gabor/wavelet studies)

## Bonus: Image Preprocessing

Includes `image_preprocessing.py` for batch cropping and normalizing TIFF microscopy images.

---

**Questions?** Open an issue on GitHub.
