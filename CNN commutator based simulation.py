import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import datetime
import Convolution
import itertools
import tensorflow as tf
import keras
import pandas as pd
import matplotlib

# Try Qt5Agg backend; fall back to Agg if display fails
try:
    matplotlib.use('Qt5Agg')
except:
    print("Qt5Agg backend failed, falling back to Agg (non-interactive, saving to file)")
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

layers = keras.layers
models = keras.models
callbacks1 = keras.callbacks
ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import cv2
import seaborn as sns
import pandas as pd
import os
import shutil
from pathlib import Path
import seaborn as sns

time_stamp = ""


# Keep the existing functions from your code
def create_gaussian_2d(size, mu, sigma_x, sigma_y, amplitude=1.0):
    """Create a 2D Gaussian matrix with different x and y spreads"""
    x = np.linspace(-size, size, size)  # Centered coordinate system
    y = np.linspace(-size, size, size)
    x, y = np.meshgrid(x, y)

    gaussian = amplitude * np.exp(
        -(((x - mu[0]) ** 2) / (2 * sigma_x ** 2) +
          ((y - mu[1]) ** 2) / (2 * sigma_y ** 2)))

    # Add random noise scaled by square root of intensity
    noise = np.random.random(gaussian.shape) * np.sqrt(gaussian)
    gaussian += noise

    return gaussian


def calculate_overlap(X, Y):
    """Calculate the normalized overlap between two matrices"""
    X_norm = X / np.sum(X)
    Y_norm = Y / np.sum(Y)
    overlap = np.sum(X_norm * Y_norm) / np.sqrt(np.sum(X_norm ** 2) * np.sum(Y_norm ** 2))
    return overlap


def calculate_stats(matrix):
    """Calculate mean and variance of the matrix"""
    mean_val = np.mean(matrix)
    var_val = np.var(matrix)
    return mean_val, var_val


def calculate_commutator(X, Y):
    """Calculate the commutator XY-YX and its Frobenius norm"""
    commutator = np.matmul(X, Y) - np.matmul(Y, X)
    norm = np.linalg.norm(commutator, 'fro')
    return commutator, norm


def plot_average_commutator(avg_commutator, norm, num_cases, output_path, size, overlap_threshold=0.7):
    """Plot the average commutator from high overlap cases"""
    plt.figure(figsize=(10, 8))

    x_ticks = np.linspace(-size, size, 9)  # 9 ticks for readability

    plt.imshow(avg_commutator, extent=[-size, size, -size, size])
    plt.colorbar(label='Amplitude')
    plt.title(f'Average Commutator from {num_cases} Cases\nwith Overlap > {overlap_threshold}\nNorm: {norm:.2e}')

    plt.xlabel('Position (arb. units)')
    plt.ylabel('Position (arb. units)')
    plt.xticks(x_ticks)
    plt.yticks(x_ticks)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_and_save(X, Y, commutator, norm, title, output_path, size):
    """Plot the matrices, commutator, and combined peaks, then save to file"""
    fig = plt.figure(figsize=(15, 10))

    gs = plt.GridSpec(2, 2, figure=fig)
    x_ticks = np.linspace(-size, size, 9)  # 9 ticks for readability
    extent = [-size, size, -size, size]

    # Plot X with colorbar and axes
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(X, extent=extent)
    ax1.set_title('Matrix X (Fixed)')
    plt.colorbar(im1, ax=ax1)
    ax1.set_xlabel('Position (arb. units)')
    ax1.set_ylabel('Position (arb. units)')
    ax1.set_xticks(x_ticks)
    ax1.set_yticks(x_ticks)

    # Plot Y with colorbar and axes
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(Y, extent=extent)
    ax2.set_title('Matrix Y (Variable)')
    plt.colorbar(im2, ax=ax2)
    ax2.set_xlabel('Position (arb. units)')
    ax2.set_ylabel('Position (arb. units)')
    ax2.set_xticks(x_ticks)
    ax2.set_yticks(x_ticks)

    # Plot commutator with colorbar and axes
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(commutator, extent=extent)
    ax3.set_title(f'Commutator (Norm: {norm:.2e})')
    plt.colorbar(im3, ax=ax3)
    ax3.set_xlabel('Position (arb. units)')
    ax3.set_ylabel('Position (arb. units)')
    ax3.set_xticks(x_ticks)
    ax3.set_yticks(x_ticks)

    # Plot combined peaks with colorbar and axes
    ax4 = fig.add_subplot(gs[1, 1])
    im4 = ax4.imshow(X + Y, extent=extent)
    ax4.set_title('Combined Peaks')
    plt.colorbar(im4, ax=ax4)
    ax4.set_xlabel('Position (arb. units)')
    ax4.set_ylabel('Position (arb. units)')
    ax4.set_xticks(x_ticks)
    ax4.set_yticks(x_ticks)

    plt.suptitle(title)
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()


def analyze_case(X, Y, case_name, stats_file, output_dir, case_number, size, results_list=None):
    """Analyze a specific case and save results"""
    Y_mean, Y_var = calculate_stats(Y)
    commutator, norm = calculate_commutator(X, Y)
    overlap = calculate_overlap(X, Y)

    title = f"{case_name} (Overlap: {overlap:.3f})"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{case_number}_overlap_{overlap:.3f}.png"
    plot_and_save(X, Y, commutator, norm, title, output_path, size)

    with open(stats_file, 'a') as f:
        f.write(f"\n{case_name}\n")
        f.write(f"Overlap: {overlap:.3f}\n")
        f.write(f"Matrix Y - Mean: {Y_mean:.6f}, Variance: {Y_var:.6f}\n")
        f.write(f"Commutator Norm: {norm:.6e}\n\n")

    # Store results for Excel export if results_list is provided
    if results_list is not None:
        results_list.append({
            'case_name': case_name,
            'case_number': case_number,
            'overlap': overlap,
            'commutator_norm': norm,
            'y_mean': Y_mean,
            'y_variance': Y_var,
            'commutator': commutator
        })

    return overlap, commutator


# New functions for CNN classifier
def generate_training_data(base_X, size, base_sigma, overlap_threshold=0.7, num_samples=500):
    """Generate training data for the CNN classifier"""
    X_data = []
    y_data = []

    # Generate samples with varying distances and scales
    for _ in range(num_samples):
        # Randomly choose distance and scale
        distance = np.random.uniform(0, 120)
        scale = np.random.uniform(0.1, 3.0)

        # Create Y with these parameters
        Y = create_gaussian_2d(size, (distance, 0), base_sigma * scale, base_sigma * scale)

        # Calculate commutator and overlap
        commutator, _ = calculate_commutator(base_X, Y)
        overlap = calculate_overlap(base_X, Y)

        # Normalize commutator for CNN input
        normalized_commutator = (commutator - np.min(commutator)) / (np.max(commutator) - np.min(commutator) + 1e-10)

        # Add to dataset
        X_data.append(normalized_commutator)
        y_data.append(1 if overlap > overlap_threshold else 0)  # 1 for similar, 0 for dissimilar

    return np.array(X_data), np.array(y_data)


def preprocess_data_for_cnn(X_data, y_data):
    """Preprocess the commutator data for CNN input"""
    # Reshape data for CNN input (add channel dimension)
    X_data = X_data.reshape(X_data.shape[0], X_data.shape[1], X_data.shape[2], 1)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val


def build_cnn_model(input_shape):
    """Build the CNN model for commutator classification"""
    model = models.Sequential([
        # First convolutional layer
        layers.Conv2D(32, (5, 5), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        # Second convolutional layer
        layers.Conv2D(64, (5, 5), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Third convolutional layer
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary classification: similar/dissimilar
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def train_model(model, X_train, y_train, X_val, y_val, output_dir):
    """Train the CNN model"""
    # Create callbacks
    callbacks = [
        callbacks1.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        callbacks1.ModelCheckpoint(str(output_dir / 'best_model.h5'), save_best_only=True)
    ]

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    return history


def evaluate_model(model, X_val, y_val, output_dir):
    """Evaluate the model and plot results"""
    # Evaluate the model
    loss, accuracy = model.evaluate(X_val, y_val)

    # Make predictions
    y_pred_prob = model.predict(X_val)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Create confusion matrix
    # cm = confusion_matrix(y_val, y_pred)

    # Plot confusion matrix
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    # plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.3f}')
    # plt.ylabel('True Label')
    # plt.xlabel('Predicted Label')
    # plt.savefig(output_dir / 'confusion_matrix.png')
    # plt.close()

    # Create classification report
    report = classification_report(y_val, y_pred)

    # Save report to file
    # with open(output_dir / 'classification_report.txt', 'w') as f:
    #     f.write(f"Model Evaluation\n")
    #     f.write(f"===============\n")
    #     f.write(f"Loss: {loss:.4f}\n")
    #     f.write(f"Accuracy: {accuracy:.4f}\n\n")
    #     f.write("Classification Report:\n")
    #     f.write(report)

    return y_pred, accuracy


def plot_learning_curves(history, output_dir):
    """Plot training and validation learning curves"""
    plt.figure(figsize=(12, 5))

    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curves.png')
    plt.close()


def visualize_predictions(X_val, y_val, y_pred, output_dir, num_examples=5):
    """Visualize some examples of commutators and model predictions"""
    # Find indices of correctly and incorrectly classified samples
    correct_indices = np.where(y_val == y_pred)[0]
    incorrect_indices = np.where(y_val != y_pred)[0]

    # Function to plot a set of examples
    def plot_examples(indices, title_prefix):
        if len(indices) == 0:
            return

        n = min(num_examples, len(indices))
        fig, axes = plt.subplots(1, n, figsize=(n * 4, 4))

        for i, idx in enumerate(indices[:n]):
            if n == 1:
                ax = axes
            else:
                ax = axes[i]

            # Get the commutator image
            commutator = X_val[idx].squeeze()

            # Plot the image
            im = ax.imshow(commutator)
            ax.set_title(f"{title_prefix}\nTrue: {y_val[idx]}, Pred: {y_pred[idx]}")
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / f'{title_prefix.lower().replace(" ", "_")}_examples.png')
        plt.close()

    # Plot examples
    plot_examples(correct_indices, "Correctly Classified")
    plot_examples(incorrect_indices, "Incorrectly Classified")


# def visualize_commutator_patterns(model, output_dir, size=300, base_sigma=20):
#     """Visualize what commutator patterns the model has learned to recognize"""
#     # Create a base X gaussian
#     center = (0, 0)
#     base_X = create_gaussian_2d(size, center, base_sigma, base_sigma)
#
#     # Create test cases with different distances and scales
#     distances = [0, 10, 20, 40, 80]
#     scales = [0.5, 1.0, 1.5, 2.0]
#     test_cases = []
#
#     for distance in distances:
#         for scale in scales:
#             Y = create_gaussian_2d(size, (distance, 0), base_sigma * scale, base_sigma * scale)
#             commutator, norm = calculate_commutator(base_X, Y)
#             overlap = calculate_overlap(base_X, Y)
#
#             # Normalize commutator for CNN input
#             normalized_commutator = (commutator - np.min(commutator)) / (
#                     np.max(commutator) - np.min(commutator) + 1e-10)
#
#             test_cases.append({
#                 'distance': distance,
#                 'scale': scale,
#                 'commutator': normalized_commutator,
#                 'overlap': overlap
#             })
#
#     # Create a figure to display the results - 5 rows, 4 columns
#     fig, axes = plt.subplots(5, 4, figsize=(16, 20))
#
#     # Process each test case
#     for i, case in enumerate(test_cases):
#         row = i // 4
#         col = i % 4
#
#         # Reshape commutator for model input
#         X_input = case['commutator'].reshape(1, size, size, 1)
#
#         # Make prediction
#         prediction = model.predict(X_input, verbose=0)[0][0]
#
#         # Plot commutator
#         im = axes[row, col].imshow(case['commutator'])
#         axes[row, col].set_title(
#             f"Dist: {case['distance']}, Scale: {case['scale']}\nTrue Overlap: {case['overlap']:.3f}\nPred: {prediction:.3f}")
#         axes[row, col].axis('off')
#
#     plt.tight_layout()
#     plt.savefig(output_dir / 'commutator_pattern_analysis.png')
#     plt.close()


def load_data_from_existing_cases(high_overlap_commutators, low_overlap_commutators, size):
    """Create training data from pre-computed commutators"""
    X_data = []
    y_data = []

    # Add high overlap cases (similar signals)
    for commutator in high_overlap_commutators:
        # Normalize commutator
        normalized_commutator = (commutator - np.min(commutator)) / (np.max(commutator) - np.min(commutator) + 1e-10)
        X_data.append(normalized_commutator)
        y_data.append(1)  # 1 for similar

    # Add low overlap cases (dissimilar signals)
    for commutator in low_overlap_commutators:
        # Normalize commutator
        normalized_commutator = (commutator - np.min(commutator)) / (np.max(commutator) - np.min(commutator) + 1e-10)
        X_data.append(normalized_commutator)
        y_data.append(0)  # 0 for dissimilar

    return np.array(X_data), np.array(y_data)


def predict_all_cases_and_export(model, all_results_data, output_dir, size):
    """Predict CNN output for all cases and export to Excel"""

    # Prepare data for CNN prediction
    prediction_results = []

    for case_data in all_results_data:
        # Get commutator and normalize it
        commutator = case_data['commutator']
        normalized_commutator = (commutator - np.min(commutator)) / (np.max(commutator) - np.min(commutator) + 1e-10)

        # Reshape for CNN input
        X_input = normalized_commutator.reshape(1, size, size, 1)

        # Make prediction
        cnn_output = model.predict(X_input, verbose=0)[0][0]
        cnn_prediction = 1 if cnn_output > 0.5  else 0

        # Add to results
        prediction_results.append({
            'Case_Name': case_data['case_name'],
            'Case_Number': case_data['case_number'],
            'Overlap_Value': case_data['overlap'],
            'CNN_Output': cnn_output,
            'CNN_Prediction_Binary': cnn_prediction,
            'Commutator_Norm': case_data['commutator_norm'],
            'Y_Mean': case_data['y_mean'],
            'Y_Variance': case_data['y_variance'],
            'High_Overlap_Ground_Truth': 1 if case_data['overlap'] > 0.8 else 0
        })

    # Create DataFrame and export to Excel
    df = pd.DataFrame(prediction_results)

    # Sort by case number for better organization
    df_sorted = df.sort_values('Case_Number')

    # Export to Excel
    excel_path = output_dir / 'CNN_Results_Summary.xlsx'

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Main results sheet
        df_sorted.to_excel(writer, sheet_name='CNN_Results', index=False)

        # Summary statistics sheet
        summary_stats = {
            'Metric': [
                'Total Cases',
                'High Overlap Cases (>0.8)',
                'Low Overlap Cases (≤0.8)',
                'CNN Predicted High Similarity',
                'CNN Predicted Low Similarity',
                'Mean CNN Output (High Overlap)',
                'Mean CNN Output (Low Overlap)',
                'CNN Accuracy',
                'Mean Overlap Value',
                'Std Overlap Value'
            ],
            'Value': [
                len(df_sorted),
                len(df_sorted[df_sorted['High_Overlap_Ground_Truth'] == 1]),
                len(df_sorted[df_sorted['High_Overlap_Ground_Truth'] == 0]),
                len(df_sorted[df_sorted['CNN_Prediction_Binary'] == 1]),
                len(df_sorted[df_sorted['CNN_Prediction_Binary'] == 0]),
                df_sorted[df_sorted['High_Overlap_Ground_Truth'] == 1]['CNN_Output'].mean(),
                df_sorted[df_sorted['High_Overlap_Ground_Truth'] == 0]['CNN_Output'].mean(),
                (df_sorted['CNN_Prediction_Binary'] == df_sorted['High_Overlap_Ground_Truth']).mean(),
                df_sorted['Overlap_Value'].mean(),
                df_sorted['Overlap_Value'].std()
            ]
        }

        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)

        # Cases by category
        high_overlap_cases = df_sorted[df_sorted['High_Overlap_Ground_Truth'] == 1].copy()
        low_overlap_cases = df_sorted[df_sorted['High_Overlap_Ground_Truth'] == 0].copy()

        if not high_overlap_cases.empty:
            high_overlap_cases.to_excel(writer, sheet_name='High_Overlap_Cases', index=False)

        if not low_overlap_cases.empty:
            low_overlap_cases.to_excel(writer, sheet_name='Low_Overlap_Cases', index=False)

    print(f"Results exported to: {excel_path}")

    # Also create a visualization of CNN outputs vs overlap values
    plt.figure(figsize=(12, 8))

    # Scatter plot of CNN output vs overlap
    colors = ['red' if gt == 0 else 'blue' for gt in df_sorted['High_Overlap_Ground_Truth']]
    plt.scatter(df_sorted['Overlap_Value'], df_sorted['CNN_Output'],
                c=colors, alpha=0.6, s=50)

    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='CNN Decision Threshold')
    plt.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='Overlap Threshold')

    plt.xlabel('Overlap Value')
    plt.ylabel('CNN Output')
    plt.title('CNN Output vs Overlap Value for All Cases')
    plt.legend(['Low Overlap GT', 'High Overlap GT', 'CNN Threshold', 'Overlap Threshold'])
    plt.grid(True, alpha=0.3)

    # Add correlation coefficient
    correlation = df_sorted['Overlap_Value'].corr(df_sorted['CNN_Output'])
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))

    plt.tight_layout()
    plt.savefig(output_dir / 'CNN_vs_Overlap_Analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    return df_sorted


def cnn_main(output_dir, high_overlap_commutators, low_overlap_commutators, all_results_data, size=300):
    """Main function for CNN classifier with Excel export"""
    # Create CNN model directory
    cnn_dir = output_dir / "cnn_classifier"
    cnn_dir.mkdir(parents=True, exist_ok=True)

    # Load data from existing cases
    X_data, y_data = load_data_from_existing_cases(high_overlap_commutators, low_overlap_commutators, size)

    # If we don't have enough data, generate more
    if len(X_data) < 100:
        # Basic parameters
        base_sigma = 20
        center = (0, 0)

        # Create fixed base matrix X at center
        base_X = create_gaussian_2d(size, center, base_sigma, base_sigma)

        # Generate more training data
        X_data_gen, y_data_gen = generate_training_data(base_X, size, base_sigma, num_samples=500)

        # Combine real and generated data
        X_data = np.concatenate([X_data, X_data_gen], axis=0)
        y_data = np.concatenate([y_data, y_data_gen], axis=0)

    # Preprocess data for CNN
    X_train, X_val, y_train, y_val = preprocess_data_for_cnn(X_data, y_data)

    # Build the CNN model
    model = build_cnn_model(input_shape=(X_train.shape[1], X_train.shape[2], 1))

    # Print model summary
    model.summary()

    # Train the model
    history = train_model(model, X_train, y_train, X_val, y_val, cnn_dir)

    # Evaluate the model
    y_pred, accuracy = evaluate_model(model, X_val, y_val, cnn_dir)

    # Plot learning curves
    plot_learning_curves(history, cnn_dir)

    # Visualize predictions
    visualize_predictions(X_val, y_val, y_pred, cnn_dir)

    # Visualize commutator patterns
    # visualize_commutator_patterns(model, cnn_dir, size)

    # Save the model
    model.save(cnn_dir / 'final_model.h5')

    # NEW: Predict for all cases and export to Excel
    results_df = predict_all_cases_and_export(model, all_results_data, cnn_dir, size)

    # Save a summary of the results
    with open(cnn_dir / 'summary.txt', 'w') as f:
        f.write("CNN Classifier for Gaussian Signal Commutativity\n")
        f.write("==============================================\n\n")
        f.write(f"Number of training samples: {len(X_train)}\n")
        f.write(f"Number of validation samples: {len(X_val)}\n")
        f.write(f"Final validation accuracy: {accuracy:.4f}\n")
        f.write(f"Total analyzed cases: {len(all_results_data)}\n\n")
        f.write("Model Description:\n")
        f.write("- CNN architecture with 3 convolutional layers\n")
        f.write("- The model classifies commutators to determine if signals are similar\n")
        f.write("- Input: 2D commutator matrix (XY-YX)\n")
        f.write("- Output: Binary classification (1 for similar signals, 0 for dissimilar)\n")
        f.write("\nResults have been exported to Excel file: CNN_Results_Summary.xlsx\n")

    return model, accuracy, results_df


def main():
    # Create timestamped output directory
    current_time = datetime.now().strftime("%d-%m-%y_%H_%M")
    global time_stamp
    time_stamp = current_time
    output_dir = Path(
        r"INPUT YOUR PATH DIRECTORY HERE")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize statistics file
    stats_file = output_dir / "gaussian_statistics.txt"

    # Basic parameters
    size = 300
    base_sigma = 20  # Adjusted sigma for larger matrix
    center = (0, 0)  # Center is now at (0,0)

    # Create fixed base matrix X at center
    base_X = create_gaussian_2d(size, center, base_sigma, base_sigma)

    # Store commutators from high overlap cases and all results
    high_overlap_commutators = []
    low_overlap_commutators = []
    all_results_data = []  # NEW: Store all case results for Excel export
    overlap_threshold = 0.8

    with open(stats_file, 'w') as f:
        f.write("Statistical Analysis of Gaussian Peaks\n")
        f.write("=====================================\n")
        f.write("Note: Matrix X remains fixed at center for all analyses\n\n")

    # A. Sliding analysis
    with open(stats_file, 'a') as f:
        f.write("\nSliding Analysis (Y sliding, X fixed)\n")
        f.write("===================================\n")

    sliding_distances = [0, 1, 2, 3,4, 5,6, 7, 10,13,15, 17,20,23,25,28, 30,35,
                         37, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    sliding_matrices = []
    for i, dist in enumerate(sliding_distances):
        Y = create_gaussian_2d(size, (dist, 0), base_sigma, base_sigma)
        sliding_matrices.append((dist, Y))
        case_name = f"Sliding Case - Y Distance {dist}"
        overlap, commutator = analyze_case(base_X, Y, case_name, stats_file, output_dir / "Commutator data",
                                           f"001_{i + 1:02d}", size, all_results_data)

        if overlap > overlap_threshold:
            high_overlap_commutators.append(commutator)
        else:
            low_overlap_commutators.append(commutator)

    # B. Smearing analysis
    with open(stats_file, 'a') as f:
        f.write("\nSmearing Analysis (Y smearing, X fixed)\n")
        f.write("====================================\n")

    scale_factors = [0.1,0.15, 0.2,0.25, 0.3,0.35, 0.4,0.45, 0.5,0.55, 0.6,0.65, 0.7,0.75,
                     0.8,0.85, 0.9,1, 1.1,1.15, 1.2,1.25, 1.3,1.35, 1.4,1.45, 1.5,1.55, 1.6,1.65,
                     1.7,1.75, 1.8, 1.9, 2.0, 2.5, 2.9]
    smearing_matrices = []
    for i, scale in enumerate(scale_factors):
        Y = create_gaussian_2d(size, center, base_sigma * scale, base_sigma * scale)
        smearing_matrices.append((scale, Y))
        case_name = f"Smearing Case - Y Scale {scale}x"
        overlap, commutator = analyze_case(base_X, Y, case_name, stats_file, output_dir / "Commutator data",
                                           f"002_{i + 1:02d}",
                                           size, all_results_data)

        if overlap > overlap_threshold:
            high_overlap_commutators.append(commutator)
        else:
            low_overlap_commutators.append(commutator)

    # C. Combinations of sliding and smearing
    with open(stats_file, 'a') as f:
        f.write("\nSliding-Smearing Combinations (Y varying, X fixed)\n")
        f.write("=============================================\n")

    combo_counter = 1
    for (dist, _) in sliding_matrices:
        for (scale, _) in smearing_matrices:
            Y = create_gaussian_2d(size,
                                   (dist, 0),
                                   base_sigma * scale, base_sigma * scale)
            case_name = f"Combination - Y Sliding {dist} with Y Smearing {scale}x"
            overlap, commutator = analyze_case(base_X, Y, case_name, stats_file, output_dir / "Commutator data",
                                               f"003_{combo_counter:02d}", size, all_results_data)

            if overlap > overlap_threshold:
                high_overlap_commutators.append(commutator)
            else:
                low_overlap_commutators.append(commutator)
            combo_counter += 1

    # Calculate and plot average commutator from high overlap cases
    if high_overlap_commutators:
        avg_commutator = np.mean(high_overlap_commutators, axis=0)
        avg_norm = np.linalg.norm(avg_commutator, 'fro')

        # Plot average commutator
        avg_output = output_dir / "average_high_overlap_commutator.png"
        plot_average_commutator(avg_commutator, avg_norm, len(high_overlap_commutators), avg_output, size)

        # Add to stats file
        with open(stats_file, 'a') as f:
            f.write("\nAverage Commutator Analysis\n")
            f.write("=========================\n")
            f.write(f"Number of cases with overlap > {overlap_threshold}: {len(high_overlap_commutators)}\n")
            f.write(f"Average commutator norm: {avg_norm:.6e}\n")

    # Train and evaluate CNN model with Excel export
    with open(stats_file, 'a') as f:
        f.write("\nCNN Classifier Analysis\n")
        f.write("======================\n")

    model, accuracy, results_df = cnn_main(output_dir, high_overlap_commutators, low_overlap_commutators,
                                           all_results_data, size)

    # Add CNN results to stats file
    with open(stats_file, 'a') as f:
        f.write(f"CNN classifier accuracy: {accuracy:.4f}\n")
        f.write(f"Total cases analyzed: {len(all_results_data)}\n")
        f.write("CNN classifier has been trained to detect similar signals based on commutator patterns.\n")
        f.write("Check the 'cnn_classifier' folder for detailed results and visualizations.\n")
        f.write("Excel summary with CNN outputs has been created: CNN_Results_Summary.xlsx\n")

    print(f"Analysis complete! Results saved to: {output_dir}")
    print(f"Excel file with CNN outputs created: {output_dir / 'cnn_classifier' / 'CNN_Results_Summary.xlsx'}")


if __name__ == "__main__":
    main()