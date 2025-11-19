import os
import numpy as np
from PIL import Image
from datetime import datetime
import matplotlib

matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import random
from scipy import ndimage

# Input directory (where the cropped images are)
input_dir = r"INPUT YOUR PATH DIRECTORY HERE"

# Output directory for commutators with timestamp
base_output_dir = r"INPUT YOUR PATH DIRECTORY HERE"
output_dir = os.path.join(base_output_dir)
output_dir_pairs = os.path.join(base_output_dir, "X_Commutator_Pairs")
output_dir_augmented = os.path.join(base_output_dir, "Augmented")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir_pairs, exist_ok=True)
os.makedirs(output_dir_augmented, exist_ok=True)
print(f"Output directory: {output_dir}")
print(f"Pairs directory: {output_dir_pairs}")
print(f"Augmented directory: {output_dir_augmented}")

# ==================== AUGMENTATION CONFIGURATION ====================
# Number of augmented versions per original commutator
NUM_AUGMENTATIONS = 20  # Increased from 5 to 20 for bigger dataset

# Augmentation probabilities (each augmentation has these chances)
AUG_FLIP_HORIZONTAL = 0.7  # Increased from 0.5
AUG_FLIP_VERTICAL = 0.7  # Increased from 0.5
AUG_ROTATE_90 = 0.4
AUG_ROTATE_180 = 0.4
AUG_ROTATE_270 = 0.4
AUG_SALT_PEPPER = 0.5  # Increased from 0.4

# Enhanced blur augmentations - multiple types and strengths
AUG_GAUSSIAN_BLUR_LIGHT = 0.20  # Very light blur (sigma=0.3-0.5)
AUG_GAUSSIAN_BLUR_MEDIUM = 0.15  # Medium blur (sigma=0.6-0.8)
AUG_GAUSSIAN_BLUR_HEAVY = 0.08  # Heavier blur (sigma=0.9-1.2)
AUG_MOTION_BLUR = 0.10  # Motion blur (directional)
AUG_DEFOCUS_BLUR = 0.08  # Defocus blur (circular)

# Crop settings - remove gray padding areas
CROP_GRAY_AREAS = True
GRAY_THRESHOLD = 10  # Pixels with value < this are considered black/gray padding

print("\n" + "=" * 80)
print("AUGMENTATION SETTINGS")
print("=" * 80)
print(f"Augmentations per commutator: {NUM_AUGMENTATIONS}")
print(f"Augmentation probabilities:")
print(f"  - Horizontal flip: {AUG_FLIP_HORIZONTAL}")
print(f"  - Vertical flip: {AUG_FLIP_VERTICAL}")
print(f"  - Rotate 90°: {AUG_ROTATE_90}")
print(f"  - Rotate 180°: {AUG_ROTATE_180}")
print(f"  - Rotate 270°: {AUG_ROTATE_270}")
print(f"  - Salt & pepper noise: {AUG_SALT_PEPPER}")
print(f"\n🌫️  Blur augmentations:")
print(f"  - Gaussian blur (light): {AUG_GAUSSIAN_BLUR_LIGHT}")
print(f"  - Gaussian blur (medium): {AUG_GAUSSIAN_BLUR_MEDIUM}")
print(f"  - Gaussian blur (heavy): {AUG_GAUSSIAN_BLUR_HEAVY}")
print(f"  - Motion blur: {AUG_MOTION_BLUR}")
print(f"  - Defocus blur: {AUG_DEFOCUS_BLUR}")
print(
    f"  - Total blur probability: ~{AUG_GAUSSIAN_BLUR_LIGHT + AUG_GAUSSIAN_BLUR_MEDIUM + AUG_GAUSSIAN_BLUR_HEAVY + AUG_MOTION_BLUR + AUG_DEFOCUS_BLUR:.2f}")
print(f"\n✂️  Cropping settings:")
print(f"  - Remove gray padding: {CROP_GRAY_AREAS}")
print(f"  - Gray threshold: {GRAY_THRESHOLD}")
print(f"\n📊 Expected dataset size:")
print(f"  - Original: ~40 commutators")
print(f"  - Augmented: ~{40 * NUM_AUGMENTATIONS} commutators")
print(f"  - Total: ~{40 * (NUM_AUGMENTATIONS + 1)} images ({NUM_AUGMENTATIONS + 1}x multiplier)")
print("=" * 80)

# Define good scans (all with 'g' prefix)
good_scans = [
    ('g', '1', '54'), ('g', '1', '55'), ('g', '1', '56'),
    ('g', '2', '54'), ('g', '2', '55'),
    ('g', '3', '54'), ('g', '3', '55'), ('g', '3', '56'), ('g', '3', '57'),
    ('g', '4', '54'), ('g', '4', '55'), ('g', '4', '56'), ('g', '4', '57'),
    ('g', '5', '54'), ('g', '5', '55'), ('g', '5', '56'), ('g', '5', '57'),
    ('g', '6', '54'), ('g', '6', '55'), ('g', '6', '56'), ('g', '6', '57'),
]

# Define bad scans (all with 'b' prefix)
bad_scans = [
    ('b', '1', '54'), ('b', '1', '55'), ('b', '1', '56'), ('b', '1', '57'),
    ('b', '2', '54'), ('b', '2', '55'), ('b', '2', '56'), ('b', '2', '57'),
    ('b', '3', '54'), ('b', '3', '55'), ('b', '3', '56'), ('b', '3', '57'),
    ('b', '4', '54'), ('b', '4', '55'), ('b', '4', '56'), ('b', '4', '57'),
    ('b', '5', '54'), ('b', '5', '55'), ('b', '5', '56'), ('b', '5', '57'),
    ('b', '6', '54'), ('b', '6', '55'), ('b', '6', '56'), ('b', '6', '57'),
]


def load_image(color, number, z_value):
    """Load image and convert to float array"""
    filename = f"RDA_{color}{number}_z{z_value}.tiff"
    filepath = os.path.join(input_dir, filename)
    try:
        img = Image.open(filepath)
        return np.array(img, dtype=np.float64)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None


def compute_commutator(X, Y):
    """Compute commutator: XY - YX using matrix multiplication with zero padding to make square"""
    # Get dimensions
    h, w = X.shape

    # Determine size for square matrix (use max dimension)
    size = max(h, w)

    # Create square matrices with zero padding
    X_square = np.zeros((size, size), dtype=np.float64)
    Y_square = np.zeros((size, size), dtype=np.float64)

    # Place original data in top-left corner
    X_square[:h, :w] = X
    Y_square[:h, :w] = Y

    # Compute matrix multiplication
    XY = np.matmul(X_square, Y_square)
    YX = np.matmul(Y_square, X_square)

    # Return commutator (full square result)
    return XY - YX


def add_salt_pepper_noise(image, amount=0.02):
    """Add salt and pepper noise to image"""
    noisy = image.copy()

    # Salt (white pixels)
    num_salt = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords[0], coords[1]] = image.max()

    # Pepper (black pixels)
    num_pepper = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords[0], coords[1]] = image.min()

    return noisy


def add_gaussian_noise(image, mean=0, sigma=0.05):
    """Add Gaussian noise to image"""
    # Calculate noise based on image range
    img_range = image.max() - image.min()
    noise = np.random.normal(mean, sigma * img_range, image.shape)
    noisy = image + noise
    return noisy


def apply_gaussian_blur(image, sigma_range=(0.3, 0.5)):
    """
    Apply Gaussian blur with random sigma in the given range

    Args:
        image: Input image
        sigma_range: Tuple of (min_sigma, max_sigma)
    """
    from scipy.ndimage import gaussian_filter
    sigma = random.uniform(sigma_range[0], sigma_range[1])
    blurred = gaussian_filter(image, sigma=sigma)
    return blurred


def apply_motion_blur(image, kernel_size=9):
    """
    Apply motion blur in a random direction
    Simulates camera movement or object motion

    Args:
        image: Input image
        kernel_size: Size of the motion blur kernel
    """
    # Create motion blur kernel in random direction
    kernel = np.zeros((kernel_size, kernel_size))

    # Random angle for motion direction
    angle = random.uniform(0, 180)

    if angle < 45 or angle > 135:
        # Horizontal-ish motion
        kernel[kernel_size // 2, :] = np.ones(kernel_size)
    else:
        # Vertical-ish motion
        kernel[:, kernel_size // 2] = np.ones(kernel_size)

    kernel = kernel / kernel.sum()

    # Apply convolution
    from scipy.ndimage import convolve
    blurred = convolve(image, kernel, mode='reflect')

    return blurred


def apply_defocus_blur(image, radius=2):
    """
    Apply defocus blur (circular/disk blur)
    Simulates out-of-focus effect

    Args:
        image: Input image
        radius: Radius of the disk blur kernel
    """
    from scipy.ndimage import uniform_filter

    # Apply uniform (box) filter which approximates disk blur
    kernel_size = 2 * radius + 1
    blurred = uniform_filter(image, size=kernel_size, mode='reflect')

    return blurred


def apply_random_blur(image):
    """
    Apply one type of blur randomly based on probabilities
    Returns blurred image and blur type description
    """
    rand = random.random()
    cumulative = 0

    # Light Gaussian blur
    cumulative += AUG_GAUSSIAN_BLUR_LIGHT
    if rand < cumulative:
        blurred = apply_gaussian_blur(image, sigma_range=(0.3, 0.5))
        return blurred, "blur_light"

    # Medium Gaussian blur
    cumulative += AUG_GAUSSIAN_BLUR_MEDIUM
    if rand < cumulative:
        blurred = apply_gaussian_blur(image, sigma_range=(0.6, 0.8))
        return blurred, "blur_med"

    # Heavy Gaussian blur
    cumulative += AUG_GAUSSIAN_BLUR_HEAVY
    if rand < cumulative:
        blurred = apply_gaussian_blur(image, sigma_range=(0.9, 1.2))
        return blurred, "blur_heavy"

    # Motion blur
    cumulative += AUG_MOTION_BLUR
    if rand < cumulative:
        kernel_size = random.choice([7, 9, 11])  # Random kernel size
        blurred = apply_motion_blur(image, kernel_size=kernel_size)
        return blurred, f"blur_motion{kernel_size}"

    # Defocus blur
    cumulative += AUG_DEFOCUS_BLUR
    if rand < cumulative:
        radius = random.randint(2, 3)  # Random radius
        blurred = apply_defocus_blur(image, radius=radius)
        return blurred, f"blur_defocus{radius}"

    # No blur applied
    return None, None


def crop_gray_padding(image, threshold=10):
    """
    Remove gray/black padding areas from the image by cropping to content
    Returns cropped image containing only the relevant data
    """
    # Find rows and columns that contain significant data (above threshold)
    row_has_data = np.any(image > threshold, axis=1)
    col_has_data = np.any(image > threshold, axis=0)

    # Find the bounding box of the data
    rows_with_data = np.where(row_has_data)[0]
    cols_with_data = np.where(col_has_data)[0]

    if len(rows_with_data) == 0 or len(cols_with_data) == 0:
        # If no data found, return original
        return image

    # Crop to the bounding box
    row_min, row_max = rows_with_data[0], rows_with_data[-1] + 1
    col_min, col_max = cols_with_data[0], cols_with_data[-1] + 1

    cropped = image[row_min:row_max, col_min:col_max]

    return cropped


def augment_commutator(commutator):
    """
    Apply random augmentations to a commutator
    Returns augmented commutator and description of augmentations applied
    """
    augmented = commutator.copy()
    aug_description = []

    # Horizontal flip
    if random.random() < AUG_FLIP_HORIZONTAL:
        augmented = np.fliplr(augmented)
        aug_description.append("hflip")

    # Vertical flip
    if random.random() < AUG_FLIP_VERTICAL:
        augmented = np.flipud(augmented)
        aug_description.append("vflip")

    # Rotation (only apply one rotation)
    rotation_choice = random.random()
    if rotation_choice < AUG_ROTATE_90:
        augmented = np.rot90(augmented, k=1)
        aug_description.append("rot90")
    elif rotation_choice < AUG_ROTATE_90 + AUG_ROTATE_180:
        augmented = np.rot90(augmented, k=2)
        aug_description.append("rot180")
    elif rotation_choice < AUG_ROTATE_90 + AUG_ROTATE_180 + AUG_ROTATE_270:
        augmented = np.rot90(augmented, k=3)
        aug_description.append("rot270")

    # Salt and pepper noise
    if random.random() < AUG_SALT_PEPPER:
        augmented = add_salt_pepper_noise(augmented, amount=0.02)
        aug_description.append("sp")

    # Apply blur (one type randomly based on probabilities)
    total_blur_prob = (AUG_GAUSSIAN_BLUR_LIGHT + AUG_GAUSSIAN_BLUR_MEDIUM +
                       AUG_GAUSSIAN_BLUR_HEAVY + AUG_MOTION_BLUR + AUG_DEFOCUS_BLUR)

    if random.random() < total_blur_prob:
        blurred, blur_desc = apply_random_blur(augmented)
        if blurred is not None:
            augmented = blurred
            aug_description.append(blur_desc)

    # Create description string
    if len(aug_description) == 0:
        aug_description.append("orig")

    description_str = "_".join(aug_description)

    return augmented, description_str


def save_commutator(commutator, name, subdir=""):
    """Save commutator result as image"""
    # Normalize for visualization
    comm_min = commutator.min()
    comm_max = commutator.max()
    if comm_max > comm_min:
        normalized = ((commutator - comm_min) / (comm_max - comm_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(commutator, dtype=np.uint8)

    if subdir:
        output_path = os.path.join(subdir, f"{name}.tiff")
    else:
        output_path = os.path.join(output_dir, f"{name}.tiff")

    img = Image.fromarray(normalized)
    img.save(output_path, compression='tiff_deflate')


def save_X_averaged(X):
    """Save the averaged X matrix"""
    # Normalize for visualization
    X_min = X.min()
    X_max = X.max()
    if X_max > X_min:
        normalized = ((X - X_min) / (X_max - X_min) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(X, dtype=np.uint8)

    output_path = os.path.join(output_dir, "X_averaged_random_selection.tiff")
    img = Image.fromarray(normalized)
    img.save(output_path, compression='tiff_deflate')
    print(f"Saved X: X_averaged_random_selection.tiff")


def save_X_and_commutator_pair(X, Y, commutator, name, y_description):
    """Save X, Y, and commutator side by side in one image with titles"""
    # Normalize X
    X_min = X.min()
    X_max = X.max()
    if X_max > X_min:
        X_normalized = ((X - X_min) / (X_max - X_min) * 255).astype(np.uint8)
    else:
        X_normalized = np.zeros_like(X, dtype=np.uint8)

    # Normalize Y
    Y_min = Y.min()
    Y_max = Y.max()
    if Y_max > Y_min:
        Y_normalized = ((Y - Y_min) / (Y_max - Y_min) * 255).astype(np.uint8)
    else:
        Y_normalized = np.zeros_like(Y, dtype=np.uint8)

    # Normalize commutator
    comm_min = commutator.min()
    comm_max = commutator.max()
    if comm_max > comm_min:
        comm_normalized = ((commutator - comm_min) / (comm_max - comm_min) * 255).astype(np.uint8)
    else:
        comm_normalized = np.zeros_like(commutator, dtype=np.uint8)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot X
    axes[0].imshow(X_normalized, cmap='gray')
    axes[0].set_title('X - Averaged Good Scans', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Plot Y
    axes[1].imshow(Y_normalized, cmap='gray')
    axes[1].set_title(f'Y - {y_description}', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Plot Commutator
    axes[2].imshow(comm_normalized, cmap='gray')
    axes[2].set_title('Commutator: XY - YX', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()

    # Save
    output_path = os.path.join(output_dir_pairs, f"Pair_{name}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# Main processing
print("\n" + "=" * 80)
print("COMMUTATOR ANALYSIS WITH ENHANCED BLUR AUGMENTATION")
print("=" * 80)

print("\nLoading good scans...")
good_images = []
for color, number, z_value in good_scans:
    img = load_image(color, number, z_value)
    if img is not None:
        good_images.append((img, color, number, z_value))
        print(f"Loaded: g{number}_z{z_value}")

print(f"\nTotal good scans loaded: {len(good_images)}")

# Load bad scans
print("\nLoading bad scans...")
bad_images = []
for color, number, z_value in bad_scans:
    img = load_image(color, number, z_value)
    if img is not None:
        bad_images.append((img, color, number, z_value))
        print(f"Loaded: b{number}_z{z_value}")

print(f"Total bad scans loaded: {len(bad_images)}")

# Organize good scans by chip number
good_by_chip = {}
for img, color, number, z_value in good_images:
    if number not in good_by_chip:
        good_by_chip[number] = []
    good_by_chip[number].append((img, color, number, z_value))

print("\n" + "=" * 80)
print("GOOD CHIPS DISTRIBUTION")
print("=" * 80)
for chip_num in sorted(good_by_chip.keys()):
    z_locs = [scan[3] for scan in good_by_chip[chip_num]]
    print(f"Chip g{chip_num}: {len(z_locs)} scans at z-locations {z_locs}")

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

print("\n" + "=" * 80)
print("PERFORMING RANDOM SELECTION")
print("=" * 80)

# Randomly select one z-location per chip
selected_scans = []
selection_details = []

for chip_num in sorted(good_by_chip.keys()):
    available_scans = good_by_chip[chip_num]
    if available_scans:
        selected_scan = random.choice(available_scans)
        selected_scans.append(selected_scan)
        selection_details.append(f"g{selected_scan[2]}_z{selected_scan[3]}")
        print(f"  Chip g{chip_num}: Selected z{selected_scan[3]}")

print(f"\n✓ Selected {len(selected_scans)} scans for averaging")

# Create detailed selection file
selection_file = os.path.join(output_dir, "selected_scans.txt")
with open(selection_file, 'w') as f:
    f.write(f"SELECTED SCANS FOR AVERAGING\n")
    f.write("=" * 60 + "\n")
    for detail in selection_details:
        f.write(f"{detail}\n")

# Average the selected good scans
selected_images = [scan[0] for scan in selected_scans]
X = np.mean(selected_images, axis=0)
print(f"✓ Averaged {len(selected_images)} good scans to create reference X")

# Save the averaged X matrix
save_X_averaged(X)

# Create a set of selected scan identifiers for quick lookup
selected_ids = set([(scan[2], scan[3]) for scan in selected_scans])

# ==================== COMPUTE ORIGINAL COMMUTATORS ====================
print("\n" + "=" * 80)
print("COMPUTING ORIGINAL COMMUTATORS")
print("=" * 80)

original_commutators = []

# 1. Compute commutators with OTHER good scans (not selected)
print(f"\nComputing commutators with non-selected good scans...")
other_good_count = 0
for img, color, number, z_value in good_images:
    if (number, z_value) not in selected_ids:
        Y = img
        chip_name = f"g{number}_z{z_value}"

        commutator = compute_commutator(X, Y)

        # Crop gray padding if enabled
        if CROP_GRAY_AREAS:
            commutator = crop_gray_padding(commutator, threshold=GRAY_THRESHOLD)

        comm_name = f"Comm_vs_{chip_name}_GOOD"
        save_commutator(commutator, comm_name)

        y_desc = f"Good Chip {chip_name}"
        save_X_and_commutator_pair(X, Y, commutator, comm_name, y_desc)

        # Store for augmentation
        original_commutators.append((commutator, comm_name, 'GOOD'))
        other_good_count += 1

print(f"  ✓ Computed {other_good_count} commutators with non-selected good chips")

# 2. Compute commutators with ALL bad scans
print(f"Computing commutators with bad scans...")
bad_count = 0
for img, color, number, z_value in bad_images:
    Y = img
    chip_name = f"b{number}_z{z_value}"

    commutator = compute_commutator(X, Y)

    # Crop gray padding if enabled
    if CROP_GRAY_AREAS:
        commutator = crop_gray_padding(commutator, threshold=GRAY_THRESHOLD)

    comm_name = f"Comm_vs_{chip_name}_BAD"
    save_commutator(commutator, comm_name)

    y_desc = f"Bad Chip {chip_name}"
    save_X_and_commutator_pair(X, Y, commutator, comm_name, y_desc)

    # Store for augmentation
    original_commutators.append((commutator, comm_name, 'BAD'))
    bad_count += 1

print(f"  ✓ Computed {bad_count} commutators with bad chips")

total_original = other_good_count + bad_count
print(f"\n✓ Total original commutators: {total_original}")

# ==================== AUGMENTATION ====================
print("\n" + "=" * 80)
print("GENERATING AUGMENTED COMMUTATORS WITH ENHANCED BLUR")
print("=" * 80)
print(f"Creating {NUM_AUGMENTATIONS} augmented versions per commutator...")
print(f"Enhanced blur types: Light Gaussian, Medium Gaussian, Heavy Gaussian, Motion, Defocus")
print(f"This will take a few minutes...\n")

augmented_count = 0
blur_type_counts = {
    'blur_light': 0,
    'blur_med': 0,
    'blur_heavy': 0,
    'blur_motion': 0,
    'blur_defocus': 0,
    'no_blur': 0
}

total_to_augment = len(original_commutators) * NUM_AUGMENTATIONS

for comm_idx, (commutator, base_name, chip_type) in enumerate(original_commutators, 1):
    for aug_idx in range(NUM_AUGMENTATIONS):
        # Apply augmentation
        augmented_comm, aug_desc = augment_commutator(commutator)

        # Track blur types
        if 'blur' in aug_desc:
            for blur_key in blur_type_counts.keys():
                if blur_key in aug_desc:
                    blur_type_counts[blur_key] += 1
                    break
        else:
            blur_type_counts['no_blur'] += 1

        # Create augmented filename
        aug_name = f"{base_name}_aug{aug_idx + 1}_{aug_desc}"

        # Save augmented commutator
        save_commutator(augmented_comm, aug_name, output_dir_augmented)

        augmented_count += 1

    # Progress update every 5 commutators
    if comm_idx % 5 == 0 or comm_idx == len(original_commutators):
        print(f"  Progress: {comm_idx}/{len(original_commutators)} original commutators processed "
              f"({augmented_count}/{total_to_augment} augmented created)")

print(f"\n✓ Generated {augmented_count} augmented commutators")
print(f"\n🌫️  Blur statistics:")
print(f"  - Light Gaussian blur: {blur_type_counts['blur_light']}")
print(f"  - Medium Gaussian blur: {blur_type_counts['blur_med']}")
print(f"  - Heavy Gaussian blur: {blur_type_counts['blur_heavy']}")
print(f"  - Motion blur: {blur_type_counts['blur_motion']}")
print(f"  - Defocus blur: {blur_type_counts['blur_defocus']}")
print(f"  - No blur applied: {blur_type_counts['no_blur']}")
total_blurred = sum(v for k, v in blur_type_counts.items() if k != 'no_blur')
print(f"  - Total with blur: {total_blurred} ({100 * total_blurred / augmented_count:.1f}%)")

# ==================== FINAL SUMMARY ====================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"Original commutators: {total_original}")
print(f"  - Non-selected good scans: {other_good_count}")
print(f"  - Bad scans: {bad_count}")
print(f"\nAugmented commutators: {augmented_count}")
print(f"  - Augmentations per original: {NUM_AUGMENTATIONS}")
print(f"\nTotal dataset size: {total_original + augmented_count}")
print(f"  - Augmentation multiplier: {(total_original + augmented_count) / total_original:.1f}x")
print(f"\nOutput directories:")
print(f"  - Original commutators: {output_dir}")
print(f"  - Augmented commutators: {output_dir_augmented}")
print(f"  - Visualization pairs: {output_dir_pairs}")
print("=" * 80)

# Create summary file
summary_file = os.path.join(output_dir, "ANALYSIS_SUMMARY.txt")
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("COMMUTATOR ANALYSIS SUMMARY WITH ENHANCED BLUR AUGMENTATION\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    f.write("STRATEGY:\n")
    f.write("- For each good chip, randomly select one z-location\n")
    f.write("- Create average X from these selected scans (one per chip)\n")
    f.write("- Compute commutators for:\n")
    f.write("    1. All other good scans (not selected for average)\n")
    f.write("    2. All bad scans\n")
    f.write("- Generate augmented versions of each commutator\n\n")

    f.write("AUGMENTATION SETTINGS:\n")
    f.write(f"Augmentations per commutator: {NUM_AUGMENTATIONS}\n")
    f.write("Augmentation types:\n")
    f.write(f"  - Horizontal flip (p={AUG_FLIP_HORIZONTAL})\n")
    f.write(f"  - Vertical flip (p={AUG_FLIP_VERTICAL})\n")
    f.write(f"  - Rotate 90° (p={AUG_ROTATE_90})\n")
    f.write(f"  - Rotate 180° (p={AUG_ROTATE_180})\n")
    f.write(f"  - Rotate 270° (p={AUG_ROTATE_270})\n")
    f.write(f"  - Salt & pepper noise (p={AUG_SALT_PEPPER})\n")
    f.write(f"\nEnhanced Blur Augmentations:\n")
    f.write(f"  - Gaussian blur (light, σ=0.3-0.5) (p={AUG_GAUSSIAN_BLUR_LIGHT})\n")
    f.write(f"  - Gaussian blur (medium, σ=0.6-0.8) (p={AUG_GAUSSIAN_BLUR_MEDIUM})\n")
    f.write(f"  - Gaussian blur (heavy, σ=0.9-1.2) (p={AUG_GAUSSIAN_BLUR_HEAVY})\n")
    f.write(f"  - Motion blur (directional) (p={AUG_MOTION_BLUR})\n")
    f.write(f"  - Defocus blur (circular) (p={AUG_DEFOCUS_BLUR})\n")
    f.write(f"\nBlur Statistics:\n")
    f.write(f"  - Light Gaussian blur: {blur_type_counts['blur_light']}\n")
    f.write(f"  - Medium Gaussian blur: {blur_type_counts['blur_med']}\n")
    f.write(f"  - Heavy Gaussian blur: {blur_type_counts['blur_heavy']}\n")
    f.write(f"  - Motion blur: {blur_type_counts['blur_motion']}\n")
    f.write(f"  - Defocus blur: {blur_type_counts['blur_defocus']}\n")
    f.write(f"  - Total with blur: {total_blurred} ({100 * total_blurred / augmented_count:.1f}%)\n")

    f.write("\nPREPROCESSING:\n")
    f.write(f"  - Crop gray padding: {CROP_GRAY_AREAS}\n")
    f.write(f"  - Gray threshold: {GRAY_THRESHOLD}\n\n")

    f.write("SELECTED SCANS FOR AVERAGING:\n")
    for detail in selection_details:
        f.write(f"  {detail}\n")
    f.write("\n")

    f.write("RESULTS:\n")
    f.write(f"Original commutators: {total_original}\n")
    f.write(f"  - Non-selected good scans: {other_good_count}\n")
    f.write(f"  - Bad scans: {bad_count}\n")
    f.write(f"Augmented commutators: {augmented_count}\n")
    f.write(f"Total dataset: {total_original + augmented_count}\n")
    f.write(f"Augmentation multiplier: {(total_original + augmented_count) / total_original:.1f}x\n\n")

    f.write("CHIP INFORMATION:\n")
    for chip_num in sorted(good_by_chip.keys()):
        z_locs = [scan[3] for scan in good_by_chip[chip_num]]
        f.write(f"  Chip g{chip_num}: Available at z-locations {z_locs}\n")
    f.write(f"\nTotal good scans: {len(good_images)}\n")
    f.write(f"Total bad scans: {len(bad_images)}\n\n")

    f.write("OUTPUT STRUCTURE:\n")
    f.write(f"  {output_dir}/\n")
    f.write("    - selected_scans.txt\n")
    f.write("    - X_averaged_random_selection.tiff\n")
    f.write(f"    - Comm_vs_*_GOOD.tiff ({other_good_count} original files)\n")
    f.write(f"    - Comm_vs_*_BAD.tiff ({bad_count} original files)\n")
    f.write(f"    Augmented/ ({augmented_count} augmented files)\n")
    f.write("      - Comm_vs_*_aug1_*.tiff\n")
    f.write("      - Comm_vs_*_aug2_*.tiff\n")
    f.write("      - ...\n")

print(f"\nSummary saved to: {summary_file}")
print("\n✅ Analysis complete with enhanced blur augmentation!")