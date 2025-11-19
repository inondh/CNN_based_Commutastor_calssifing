import os
from PIL import Image
import numpy as np

# Crop coordinates
Y_CENTER = 1150
X_RANGE = [560, 1260]
Y_RANGE = [(2300 - Y_CENTER) - 200, (2300 - Y_CENTER) + 200]

# Output directory

output_dir = r"INPUT YOUR PATH DIRECTORY HERE"

# Create directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

file_list = [
    [r"./DATA/g1/g1_z54_th08_s400_part1.tif", 'g', '1', '54'],
    [r"./DATA/g1/g1_z55_th08_s400_part1.tif", 'g', '1', '55'],
    [r"./DATA/g1/g1_z56_th08_s400_part1.tif", 'g', '1', '56'],

    [r"./DATA/g2/g2_z54_th08_s400_part1.tif", 'g', '2', '54'],
    [r"./DATA/g2/g2_z55_th08_s400_part1.tif", 'g', '2', '55'],

    [r"./DATA/g3/g3_z54_th08_s400_part1.tif", 'g', '3', '54'],
    [r"./DATA/g3/g3_z55_th08_s400_part1.tif", 'g', '3', '55'],
    [r"./DATA/g3/g3_z56_th08_s400_part1.tif", 'g', '3', '56'],
    [r"./DATA/g3/g3_z57_th08_s400_part1.tif", 'g', '3', '57'],

    [r"./DATA/g4/g4_z54_th08_s400_part1.tif", 'g', '4', '54'],
    [r"./DATA/g4/g4_z55_th08_s400_part1.tif", 'g', '4', '55'],
    [r"./DATA/g4/g4_z56_th08_s400_part1.tif", 'g', '4', '56'],
    [r"./DATA/g4/g4_z57_th08_s400_part1.tif", 'g', '4', '57'],

    [r"./DATA/g5/g5_z54_th08_s400_part1.tif", 'g', '5', '54'],
    [r"./DATA/g5/g5_z55_th08_s400_part1.tif", 'g', '5', '55'],
    [r"./DATA/g5/g5_z56_th08_s400_part1.tif", 'g', '5', '56'],
    [r"./DATA/g5/g5_z57_th08_s400_part1.tif", 'g', '5', '57'],

    [r"./DATA/g6/g6_z54_th08_s400_part1.tif", 'g', '6', '54'],
    [r"./DATA/g6/g6_z55_th08_s400_part1.tif", 'g', '6', '55'],
    [r"./DATA/g6/g6_z56_th08_s400_part1.tif", 'g', '6', '56'],
    [r"./DATA/g6/g6_z57_th08_s400_part1.tif", 'g', '6', '57'],

    [r"./DATA/b1/b1_z54_th08_s400_part1.tif", 'b', '1', '54'],
    [r"./DATA/b1/b1_z55_th08_s400_part1.tif", 'b', '1', '55'],
    [r"./DATA/b1/b1_z56_th08_s400_part1.tif", 'b', '1', '56'],
    [r"./DATA/b1/b1_z57_th08_s400_part1.tif", 'b', '1', '57'],

    [r"./DATA/b2/b2_z54_th08_s400_part1.tif", 'b', '2', '54'],
    [r"./DATA/b2/b2_z55_th08_s400_part1.tif", 'b', '2', '55'],
    [r"./DATA/b2/b2_z56_th08_s400_part1.tif", 'b', '2', '56'],
    [r"./DATA/b2/b2_z57_th08_s400_part1.tif", 'b', '2', '57'],

    [r"./DATA/b3/b3_z54_th08_s400_part1.tif", 'b', '3', '54'],
    [r"./DATA/b3/b3_z55_th08_s400_part1.tif", 'b', '3', '55'],
    [r"./DATA/b3/b3_z56_th08_s400_part1.tif", 'b', '3', '56'],
    [r"./DATA/b3/b3_z57_th08_s400_part1.tif", 'b', '3', '57'],

    [r"./DATA/b4/b4_z55_th08_s400_part1.tif", 'b', '4', '55'],
    [r"./DATA/b4/b4_z54_th08_s400_part1.tif", 'b', '4', '54'],
    [r"./DATA/b4/b4_z56_th08_s400_part1.tif", 'b', '4', '56'],
    [r"./DATA/b4/b4_z57_th08_s400_part1.tif", 'b', '4', '57'],

    [r"./DATA/b5/b5_z54_th08_s400_part1.tif", 'b', '5', '54'],
    [r"./DATA/b5/b5_z55_th08_s400_part1.tif", 'b', '5', '55'],
    [r"./DATA/b5/b5_z56_th08_s400_part1.tif", 'b', '5', '56'],
    [r"./DATA/b5/b5_z57_th08_s400_part1.tif", 'b', '5', '57'],

    [r"./DATA/b6/b6_z54_th08_s400_part1.tif", 'b', '6', '54'],
    [r"./DATA/b6/b6_z55_th08_s400_part1.tif", 'b', '6', '55'],
    [r"./DATA/b6/b6_z56_th08_s400_part1.tif", 'b', '6', '56'],
    [r"./DATA/b6/b6_z57_th08_s400_part1.tif", 'b', '6', '57'],
]

# Process each file
processed_count = 0
error_count = 0

for file_info in file_list:
    file_path = file_info[0]
    color = file_info[1]  # 'g' or 'b'
    number = file_info[2]  # sample number
    z_value = file_info[3]  # z value

    try:
        # Load image
        img = Image.open(file_path)

        # Convert to numpy array
        img_array = np.array(img)

        # Crop using coordinates
        cropped = img_array[Y_RANGE[0]:Y_RANGE[1], X_RANGE[0]:X_RANGE[1]]

        # Handle floating point images with better contrast enhancement
        if cropped.dtype == np.float32 or cropped.dtype == np.float64:
            # Use percentile-based normalization for better contrast
            p2, p98 = np.percentile(cropped, (2, 98))
            cropped_normalized = np.clip((cropped - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
        else:
            cropped_normalized = cropped

        # Create filename: RDA_color_number_z.tiff (keeping TIFF format for quality)
        output_filename = f"RDA_{color}{number}_z{z_value}.tiff"
        output_path = os.path.join(output_dir, output_filename)

        # Save cropped image as TIFF with compression
        cropped_img = Image.fromarray(cropped_normalized)
        cropped_img.save(output_path, compression='tiff_deflate')

        processed_count += 1
        print(f"✓ Saved: {output_filename}")

    except Exception as e:
        error_count += 1
        print(f"✗ Error processing {file_path}: {str(e)}")

print(f"\n--- Summary ---")
print(f"Successfully processed: {processed_count}")
print(f"Errors: {error_count}")
print(f"Output directory: {output_dir}")