import os
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from keras import layers, models, callbacks as keras_callbacks, regularizers
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
import pandas as pd
import gc

# ==================== MEMORY OPTIMIZATION ====================
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# ==================== CONFIGURATION ====================
commutator_dir = r"INPUT YOUR PATH DIRECTORY HERE"
augmented_dir = os.path.join(commutator_dir, "Augmented")

IMG_SIZE = 128

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(commutator_dir) / "INPUT YOUR PATH DIRECTORY HERE" / timestamp
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("BATCH SIZE 16 CNN TRAINING - BALANCED APPROACH")
print("=" * 80)
print(f"Original commutator directory: {commutator_dir}")
print(f"Augmented data directory: {augmented_dir}")
print(f"CNN results will be saved to: {output_dir}")
print("\n🚀 OPTIMIZATIONS:")
print(f"  ✓ Image size: {IMG_SIZE}x{IMG_SIZE}")
print("  ✓ Batch size: 16 (BALANCED for stability and speed)")
print("  ✓ MINIMAL regularization: Maximum learning capacity")
print("  ✓ MINIMAL dropout: Less restriction")
print("  ✓ Class weights: Handles class imbalance")
print("  ✓ OPTIMIZED learning rate: For batch size 16")
print("  ✓ Deeper network: More capacity")
print("  ✓ Training epochs: 50 with early stopping")
print("=" * 80)


def load_commutator(filepath, target_size=(IMG_SIZE, IMG_SIZE)):
    """Load a commutator TIFF file, resize, and normalize it"""
    try:
        img = Image.open(filepath)
        commutator = np.array(img, dtype=np.float32)

        img_pil = Image.fromarray(commutator)
        img_pil = img_pil.resize(target_size, Image.BILINEAR)
        commutator = np.array(img_pil, dtype=np.float32)

        comm_min = commutator.min()
        comm_max = commutator.max()
        if comm_max > comm_min:
            normalized = (commutator - comm_min) / (comm_max - comm_min)
        else:
            normalized = np.zeros_like(commutator, dtype=np.float32)

        return normalized
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def collect_commutators(commutator_dir, augmented_dir):
    """Collect all commutators from both original and augmented directories"""
    all_metadata = []

    # ==================== ORIGINAL COMMUTATORS ====================
    print("\n" + "=" * 60)
    print("LOADING ORIGINAL COMMUTATORS")
    print("=" * 60)

    comm_files = [f for f in os.listdir(commutator_dir)
                  if f.endswith('.tiff') and f.startswith('Comm_')]

    print(f"Found {len(comm_files)} original commutator files")

    for filename in comm_files:
        filepath = os.path.join(commutator_dir, filename)
        commutator = load_commutator(filepath)

        if commutator is None:
            continue

        if '_GOOD' in filename:
            chip_type = 'GOOD'
            label = 1
        elif '_BAD' in filename:
            chip_type = 'BAD'
            label = 0
        else:
            continue

        parts = filename.split('_vs_')
        if len(parts) >= 2:
            chip_info = parts[1].replace('.tiff', '').split('_')[0:2]
            chip_id = '_'.join(chip_info) if len(chip_info) >= 2 else filename
        else:
            chip_id = filename

        metadata = {
            'filename': filename,
            'chip_id': chip_id,
            'chip_type': chip_type,
            'label': label,
            'commutator': commutator,
            'source': 'original'
        }
        all_metadata.append(metadata)

    original_count = len(all_metadata)
    print(f"Loaded {original_count} original commutators")
    good_count = sum(1 for m in all_metadata if m['chip_type'] == 'GOOD')
    bad_count = sum(1 for m in all_metadata if m['chip_type'] == 'BAD')
    print(f"  - GOOD (label=1): {good_count}")
    print(f"  - BAD (label=0): {bad_count}")

    # ==================== AUGMENTED COMMUTATORS ====================
    print("\n" + "=" * 60)
    print("LOADING AUGMENTED COMMUTATORS (THIS MAY TAKE A WHILE...)")
    print("=" * 60)

    if os.path.exists(augmented_dir):
        aug_files = [f for f in os.listdir(augmented_dir)
                     if f.endswith('.tiff') and f.startswith('Comm_')]

        print(f"Found {len(aug_files)} augmented commutator files")
        print("Loading augmented data (this will take 2-5 minutes)...")

        loaded_count = 0
        for filename in aug_files:
            filepath = os.path.join(augmented_dir, filename)
            commutator = load_commutator(filepath)

            if commutator is None:
                continue

            if '_GOOD' in filename:
                chip_type = 'GOOD'
                label = 1
            elif '_BAD' in filename:
                chip_type = 'BAD'
                label = 0
            else:
                continue

            parts = filename.split('_vs_')
            if len(parts) >= 2:
                chip_info = parts[1].split('_aug')[0]
                aug_info = filename.split('_aug')[1].replace('.tiff', '') if '_aug' in filename else 'unknown'
            else:
                chip_info = filename
                aug_info = 'unknown'

            metadata = {
                'filename': filename,
                'chip_id': chip_info,
                'chip_type': chip_type,
                'label': label,
                'commutator': commutator,
                'source': 'augmented',
                'aug_info': aug_info
            }
            all_metadata.append(metadata)

            loaded_count += 1
            if loaded_count % 100 == 0:
                print(f"  Loaded {loaded_count}/{len(aug_files)} augmented images...")

        augmented_count = len(all_metadata) - original_count
        print(f"✓ Loaded {augmented_count} augmented commutators")
        aug_good_count = sum(1 for m in all_metadata[original_count:] if m['chip_type'] == 'GOOD')
        aug_bad_count = sum(1 for m in all_metadata[original_count:] if m['chip_type'] == 'BAD')
        print(f"  - GOOD (label=1): {aug_good_count}")
        print(f"  - BAD (label=0): {aug_bad_count}")
    else:
        print(f"Augmented directory not found: {augmented_dir}")
        augmented_count = 0

    # ==================== SUMMARY ====================
    print("\n" + "=" * 60)
    print("TOTAL DATASET SUMMARY")
    print("=" * 60)
    print(f"Total commutators: {len(all_metadata)}")
    print(f"  - Original: {original_count}")
    print(f"  - Augmented: {augmented_count}")
    print(f"  - Total multiplier: {len(all_metadata) / original_count if original_count > 0 else 0:.1f}x")

    total_good = sum(1 for m in all_metadata if m['chip_type'] == 'GOOD')
    total_bad = sum(1 for m in all_metadata if m['chip_type'] == 'BAD')
    print(f"\nClass distribution:")
    print(f"  - GOOD (label=1): {total_good} ({100 * total_good / len(all_metadata):.1f}%)")
    print(f"  - BAD (label=0): {total_bad} ({100 * total_bad / len(all_metadata):.1f}%)")

    if len(all_metadata) > 1000:
        print(f"\n🎯 MASSIVE DATASET: {len(all_metadata)} images is excellent for deep learning!")

    all_comms = [m['commutator'] for m in all_metadata]
    print(f"\nData statistics:")
    print(f"  Min: {np.min(all_comms):.4f}")
    print(f"  Max: {np.max(all_comms):.4f}")
    print(f"  Mean: {np.mean(all_comms):.4f}")
    print(f"  Std: {np.std(all_comms):.4f}")

    print("=" * 60)

    return all_metadata


def create_train_test_split(all_metadata, test_size=0.2, random_state=42):
    """Create train/test split"""
    print(f"\n{'=' * 60}")
    print("TRAIN/TEST SPLIT")
    print(f"{'=' * 60}")
    print(f"Total samples: {len(all_metadata)}")
    print(f"Test ratio: {test_size}")

    X_all = np.array([item['commutator'] for item in all_metadata], dtype=np.float32)
    y_all = np.array([item['label'] for item in all_metadata])

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X_all, y_all, np.arange(len(all_metadata)),
        test_size=test_size,
        random_state=random_state,
        stratify=y_all
    )

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

    print(f"\nTraining data: {X_train.shape}")
    print(f"  - GOOD (label=1): {(y_train == 1).sum()}")
    print(f"  - BAD (label=0): {(y_train == 0).sum()}")
    print(f"Test data: {X_test.shape}")
    print(f"  - GOOD (label=1): {(y_test == 1).sum()}")
    print(f"  - BAD (label=0): {(y_test == 0).sum()}")
    print(f"{'=' * 60}\n")

    train_metadata = [all_metadata[i] for i in train_idx]
    test_metadata = [all_metadata[i] for i in test_idx]

    del X_all
    gc.collect()

    return X_train, y_train, X_test, y_test, train_metadata, test_metadata


# ==================== OPTIMIZED MODEL FOR BATCH SIZE 16 ====================
def build_aggressive_cnn(input_shape):
    """
    Optimized CNN with batch size 16 for balanced performance

    Key changes:
    - ALMOST NO L2 regularization (0.00001)
    - MINIMAL dropout
    - DEEPER network (4 blocks)
    - More filters
    - Optimized for batch size 16
    """
    inputs = layers.Input(shape=input_shape)

    # ========== BLOCK 1: Initial feature extraction (64 filters) ==========
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.00001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.1)(x)

    # ========== BLOCK 2: Intermediate features (128 filters) ==========
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.00001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.1)(x)

    # ========== BLOCK 3: High-level features (256 filters) ==========
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.00001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.15)(x)

    # ========== BLOCK 4: Abstract features (512 filters) ==========
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.00001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)

    # ========== CLASSIFICATION HEAD ==========
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.00001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.00001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    # Learning rate for batch size 16
    # LR scales with sqrt(batch_size): 0.001 * sqrt(16/8) ≈ 0.00141
    optimizer = keras.optimizers.Adam(learning_rate=0.00141)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )

    return model


def train_cnn(model, X_train, y_train, X_val, y_val, output_dir, epochs=50):
    """Train with batch size 16 for balanced performance"""
    print(f"\n{'=' * 60}")
    print(f"TRAINING WITH BATCH SIZE 16 FOR {epochs} EPOCHS")
    print(f"{'=' * 60}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Batch size: 16 (BALANCED for stability and speed)")

    from sklearn.utils.class_weight import compute_class_weight

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    # Moderate class weights for batch size 16
    class_weight_dict = {
        0: class_weights[0] * 2.0,  # Balanced 2.0x for batch size 16
        1: class_weights[1] * 2.0
    }

    print(f"\n🎯 STRONG Class weights (2.0x multiplier):")
    print(f"  Class 0 (BAD): {class_weight_dict[0]:.4f}")
    print(f"  Class 1 (GOOD): {class_weight_dict[1]:.4f}")

    checkpoint_path = str(output_dir / 'best_model.h5')
    callbacks_list = [
        keras_callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        keras_callbacks.EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=10,  # Balanced patience for batch size 16
            verbose=1,
            restore_best_weights=True
        ),
        keras_callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,  # Moderate patience
            min_lr=1e-7,
            verbose=1
        ),
        keras_callbacks.CSVLogger(
            str(output_dir / 'training_log.csv')
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=16,  # Balanced batch size
        callbacks=callbacks_list,
        class_weight=class_weight_dict,
        verbose=1
    )

    return history


def plot_learning_curves(history, output_dir):
    """Plot comprehensive learning curves"""
    metrics_to_plot = ['loss', 'accuracy', 'precision', 'recall', 'auc']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        if metric in history.history:
            axes[idx].plot(history.history[metric], label=f'Train {metric.upper()}', linewidth=2)
            if f'val_{metric}' in history.history:
                axes[idx].plot(history.history[f'val_{metric}'], label=f'Val {metric.upper()}', linewidth=2)
            axes[idx].set_xlabel('Epoch', fontsize=12)
            axes[idx].set_ylabel(metric.capitalize(), fontsize=12)
            axes[idx].set_title(f'{metric.capitalize()} over Epochs', fontsize=14, fontweight='bold')
            axes[idx].legend(fontsize=10)
            axes[idx].grid(True, alpha=0.3)

    axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig(str(output_dir / 'learning_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Learning curves saved to: {output_dir / 'learning_curves.png'}")


def evaluate_model(model, X_test, y_test, output_dir):
    """Comprehensive model evaluation"""
    print(f"\n{'=' * 60}")
    print("MODEL EVALUATION")
    print(f"{'=' * 60}")

    y_pred_prob = model.predict(X_test, batch_size=16, verbose=0).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_test, y_pred_prob)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score
    }

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc_score:.4f}")

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['BAD (0)', 'GOOD (1)'],
                yticklabels=['BAD (0)', 'GOOD (1)'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Batch Size 16 CNN', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(str(output_dir / 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, linewidth=3, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve - Batch Size 16 CNN', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_dir / 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    report = classification_report(y_test, y_pred, target_names=['BAD (0)', 'GOOD (1)'])
    print(f"\nClassification Report:\n{report}")

    with open(str(output_dir / 'classification_report.txt'), 'w') as f:
        f.write("CLASSIFICATION REPORT - BATCH SIZE 16 CNN\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)

    return y_pred_prob, y_pred, metrics, cm


def predict_and_analyze(model, test_metadata, X_test, y_test, output_dir):
    """Predict on test set and create detailed analysis"""
    print(f"\n{'=' * 60}")
    print("DETAILED PREDICTION ANALYSIS")
    print(f"{'=' * 60}")

    predictions = model.predict(X_test, batch_size=16, verbose=0).flatten()

    results = []
    for i, metadata in enumerate(test_metadata):
        result = {
            'Filename': metadata['filename'],
            'Chip_ID': metadata['chip_id'],
            'True_Type': metadata['chip_type'],
            'True_Label': metadata['label'],
            'CNN_Raw_Output': predictions[i],
            'CNN_Prediction': 'GOOD' if predictions[i] > 0.5 else 'BAD',
            'Predicted_Label': 1 if predictions[i] > 0.5 else 0,
            'Correct': (1 if predictions[i] > 0.5 else 0) == metadata['label'],
            'Confidence': predictions[i] if predictions[i] > 0.5 else 1 - predictions[i],
            'Source': metadata.get('source', 'original')
        }
        results.append(result)

    df = pd.DataFrame(results)

    excel_path = output_dir / 'detailed_predictions.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='All_Predictions', index=False)

        correct = df[df['Correct'] == True].copy()
        correct.to_excel(writer, sheet_name='Correct', index=False)

        misclassified = df[df['Correct'] == False].copy()
        misclassified.to_excel(writer, sheet_name='Misclassified', index=False)

        good_mean = df[df['True_Type'] == 'GOOD']['CNN_Raw_Output'].mean() if len(
            df[df['True_Type'] == 'GOOD']) > 0 else 0
        good_std = df[df['True_Type'] == 'GOOD']['CNN_Raw_Output'].std() if len(
            df[df['True_Type'] == 'GOOD']) > 0 else 0
        bad_mean = df[df['True_Type'] == 'BAD']['CNN_Raw_Output'].mean() if len(
            df[df['True_Type'] == 'BAD']) > 0 else 0
        bad_std = df[df['True_Type'] == 'BAD']['CNN_Raw_Output'].std() if len(df[df['True_Type'] == 'BAD']) > 0 else 0

        good_acc = df[df['True_Type'] == 'GOOD']['Correct'].mean() if len(
            df[df['True_Type'] == 'GOOD']) > 0 else 0
        bad_acc = df[df['True_Type'] == 'BAD']['Correct'].mean() if len(df[df['True_Type'] == 'BAD']) > 0 else 0

        summary = pd.DataFrame({
            'Metric': [
                'Dataset Type',
                'Image Size',
                'Batch Size',
                'Architecture',
                'Learning Rate',
                'L2 Regularization',
                'Class Weight Multiplier',
                'Max Epochs',
                'Interpretation',
                'Decision Boundary',
                'Total Test Samples',
                'GOOD Samples',
                'BAD Samples',
                'Overall Accuracy',
                'GOOD Accuracy',
                'BAD Accuracy',
                'Mean CNN Output (GOOD)',
                'Std CNN Output (GOOD)',
                'Mean CNN Output (BAD)',
                'Std CNN Output (BAD)',
                'Separation Score',
                'Misclassified Count',
                'Original Test Samples',
                'Augmented Test Samples'
            ],
            'Value': [
                'MASSIVE (20x augmentation)',
                f'{IMG_SIZE}x{IMG_SIZE}',
                '16 (BALANCED)',
                'Deep CNN (4 Blocks: 64→128→256→512)',
                '0.00141 (scaled for batch 16)',
                '0.00001',
                '2.0x',
                '50',
                'CNN >0.5 = GOOD, CNN <0.5 = BAD',
                '0.5',
                len(df),
                len(df[df['True_Type'] == 'GOOD']),
                len(df[df['True_Type'] == 'BAD']),
                df['Correct'].mean(),
                good_acc,
                bad_acc,
                good_mean,
                good_std,
                bad_mean,
                bad_std,
                abs(good_mean - bad_mean),
                len(misclassified),
                len(df[df['Source'] == 'original']),
                len(df[df['Source'] == 'augmented'])
            ]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)

    print(f"\nResults saved to: {excel_path}")

    plot_results(df, output_dir)

    return df


def plot_results(df, output_dir):
    """Plot comprehensive CNN output distributions"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    good_outputs = df[df['True_Type'] == 'GOOD']['CNN_Raw_Output']
    bad_outputs = df[df['True_Type'] == 'BAD']['CNN_Raw_Output']

    # Histogram
    if len(good_outputs) > 0:
        axes[0, 0].hist(good_outputs, bins=30, alpha=0.7, label='GOOD (should be >0.5)',
                        color='blue', edgecolor='black')
    if len(bad_outputs) > 0:
        axes[0, 0].hist(bad_outputs, bins=30, alpha=0.7, label='BAD (should be <0.5)',
                        color='red', edgecolor='black')
    axes[0, 0].axvline(x=0.5, color='black', linestyle='--', linewidth=3, label='Threshold (0.5)')
    axes[0, 0].set_xlabel('CNN Output', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('CNN Output Distribution - Batch Size 16\n(>0.5 = GOOD, <0.5 = BAD)',
                         fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    # Box plot
    data_for_box = []
    labels_for_box = []
    if len(bad_outputs) > 0:
        data_for_box.append(bad_outputs)
        labels_for_box.append('BAD\n(should be <0.5)')
    if len(good_outputs) > 0:
        data_for_box.append(good_outputs)
        labels_for_box.append('GOOD\n(should be >0.5)')

    if len(data_for_box) > 0:
        box = axes[0, 1].boxplot(data_for_box, labels=labels_for_box,
                                 patch_artist=True, widths=0.6)
        for i, patch in enumerate(box['boxes']):
            if i == 0 and len(bad_outputs) > 0:
                patch.set_facecolor('lightcoral')
            else:
                patch.set_facecolor('lightblue')
        axes[0, 1].axhline(y=0.5, color='black', linestyle='--', linewidth=3, label='Threshold')
        axes[0, 1].set_ylabel('CNN Output', fontsize=12)
        axes[0, 1].set_title('CNN Output by Type', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Violin plot
    if len(data_for_box) > 0:
        parts = axes[1, 0].violinplot(data_for_box, positions=range(len(data_for_box)),
                                      showmeans=True, showmedians=True)
        axes[1, 0].axhline(y=0.5, color='black', linestyle='--', linewidth=3, label='Threshold')
        axes[1, 0].set_xticks(range(len(labels_for_box)))
        axes[1, 0].set_xticklabels(labels_for_box)
        axes[1, 0].set_ylabel('CNN Output', fontsize=12)
        axes[1, 0].set_title('CNN Output Distribution (Violin Plot)', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Scatter plot
    good_indices = df[df['True_Type'] == 'GOOD'].index
    bad_indices = df[df['True_Type'] == 'BAD'].index

    if len(good_outputs) > 0:
        axes[1, 1].scatter(range(len(good_indices)),
                           df.loc[good_indices, 'CNN_Raw_Output'],
                           c=df.loc[good_indices, 'Correct'].map({True: 'green', False: 'red'}),
                           alpha=0.6, s=100, label='GOOD', edgecolors='black', linewidth=1)
    if len(bad_outputs) > 0:
        axes[1, 1].scatter(range(len(bad_indices)),
                           df.loc[bad_indices, 'CNN_Raw_Output'],
                           c=df.loc[bad_indices, 'Correct'].map({True: 'green', False: 'red'}),
                           alpha=0.6, s=100, label='BAD', marker='s', edgecolors='black', linewidth=1)
    axes[1, 1].axhline(y=0.5, color='black', linestyle='--', linewidth=3, label='Threshold')
    axes[1, 1].set_xlabel('Sample Index', fontsize=12)
    axes[1, 1].set_ylabel('CNN Output', fontsize=12)
    axes[1, 1].set_title('Individual Predictions\n(Green=Correct, Red=Wrong)',
                         fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_dir / 'results_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("\nStarting BATCH SIZE 16 CNN training...")
    print("\nKEY OPTIMIZATIONS:")
    print("  ✅ BATCH SIZE 16 (BALANCED for stability and speed)")
    print("  ✅ 50 EPOCHS (with patience=10)")
    print("  ✅ DEEPER network: 4 blocks (64→128→256→512)")
    print("  ✅ MINIMAL L2 regularization (0.00001)")
    print("  ✅ MINIMAL dropout (0.1-0.2)")
    print("  ✅ MODERATE learning rate (0.00141 - optimized for batch 16)")
    print("  ✅ STRONG class weights (2.0x multiplier)")
    print("  ✅ BALANCED patience (10 epochs)")

    all_metadata = collect_commutators(commutator_dir, augmented_dir)

    if len(all_metadata) == 0:
        print("\nError: No commutators found!")
        return

    if len(all_metadata) < 500:
        print(f"\n⚠️  WARNING: Only {len(all_metadata)} images found.")
        print("Expected ~1,209 images with 30x augmentation.")
        print("Did you run the MASSIVE augmentation script?")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return

    X_train, y_train, X_test, y_test, train_metadata, test_metadata = create_train_test_split(
        all_metadata, test_size=0.2, random_state=42
    )

    print("\nBuilding CNN with BATCH SIZE 16...")
    print("  - Batch size 16 for balanced training")
    print("  - Moderate learning rate (0.00141)")
    print("  - Strong class weights (2.0x)")
    print("  - Balanced patience (10 epochs)")
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = build_aggressive_cnn(input_shape)
    model.summary()

    history = train_cnn(model, X_train, y_train, X_test, y_test, output_dir, epochs=50)

    plot_learning_curves(history, output_dir)

    _, _, metrics, cm = evaluate_model(model, X_test, y_test, output_dir)

    results_df = predict_and_analyze(model, test_metadata, X_test, y_test, output_dir)

    model.save(str(output_dir / 'final_model.h5'), save_format='h5')
    print(f"\nModel saved to: {output_dir / 'final_model.h5'}")

    print("\n" + "=" * 80)
    print("BATCH SIZE 16 CNN TRAINING COMPLETE!")
    print("=" * 80)
    print("\nINTERPRETATION:")
    print("  CNN output >0.5 = Chip is GOOD")
    print("  CNN output <0.5 = Chip is BAD")
    print(f"\nTest Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.2%}")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall:    {metrics['recall']:.2%}")
    print(f"  F1 Score:  {metrics['f1']:.2%}")
    print(f"  AUC:       {metrics['auc']:.4f}")

    good_acc = results_df[results_df['True_Type'] == 'GOOD']['Correct'].mean() if len(
        results_df[results_df['True_Type'] == 'GOOD']) > 0 else 0
    bad_acc = results_df[results_df['True_Type'] == 'BAD']['Correct'].mean() if len(
        results_df[results_df['True_Type'] == 'BAD']) > 0 else 0

    print(f"\nClass-wise Accuracy:")
    print(f"  GOOD: {good_acc:.2%}")
    print(f"  BAD:  {bad_acc:.2%}")

    good_mean = results_df[results_df['True_Type'] == 'GOOD']['CNN_Raw_Output'].mean() if len(
        results_df[results_df['True_Type'] == 'GOOD']) > 0 else 0
    bad_mean = results_df[results_df['True_Type'] == 'BAD']['CNN_Raw_Output'].mean() if len(
        results_df[results_df['True_Type'] == 'BAD']) > 0 else 0

    print(f"\nCNN Output Means:")
    print(f"  GOOD: {good_mean:.4f} (target: >0.5)")
    print(f"  BAD:  {bad_mean:.4f} (target: <0.5)")
    print(f"  Separation: {abs(good_mean - bad_mean):.4f}")

    if good_mean > 0.5 and bad_mean < 0.5:
        print("\n✅ EXCELLENT: Classes are properly separated!")
    elif abs(good_mean - bad_mean) > 0.3:
        print("\n✓ GOOD: Strong separation between classes")
    elif abs(good_mean - bad_mean) > 0.15:
        print("\n⚠️  MODERATE: Acceptable separation but could be better")
    else:
        print("\n❌ WARNING: Poor separation - data may not be linearly separable")

    print(f"\nDataset composition:")
    orig_test = len(results_df[results_df['Source'] == 'original'])
    aug_test = len(results_df[results_df['Source'] == 'augmented'])
    print(f"  Test set - Original: {orig_test}, Augmented: {aug_test}")
    print(f"  Total training images used: {len(all_metadata)}")

    print(f"\nAll results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()