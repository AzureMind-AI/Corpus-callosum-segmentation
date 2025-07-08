import os
from pathlib import Path
import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm

def process_and_save_slices(nifti_dir: Path, output_dir: Path, file_type: str = "image"):
    """
    Loads NIfTI files from a directory, extracts a specific 2D slice,
    and saves it as a PNG image in the output directory.

    Args:
        nifti_dir (Path): The directory containing the source NIfTI files.
        output_dir (Path): The directory where the PNG images will be saved.
        file_type (str): A descriptor for printing progress (e.g., "image" or "label").
    """
    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing NIfTI files from '{nifti_dir}'...")
    print(f"Saving 2D PNG files to '{output_dir}'...")

    nifti_files = list(nifti_dir.glob('**/*.nii*'))
    if not nifti_files:
        print(f"Warning: No NIfTI files found in {nifti_dir}")
        return

    for nifti_path in tqdm(nifti_files, desc=f"Converting {file_type}s"):
        try:
            # Load the NIfTI file
            nifti_img = nib.load(nifti_path)
            data = nifti_img.get_fdata()

            # --- Extract the slice ---
            # We use index 10 for the 11th slice, matching the original Dataset
            slice_idx = 10 
            if data.shape[2] <= slice_idx:
                print(f"Warning: Skipping {nifti_path.name}, has only {data.shape[2]} slices, need at least {slice_idx + 1}")
                continue
            
            if data.ndim == 4:
                # 4D data (e.g., time-series or multi-modal)
                slice_2d = data[:, :, slice_idx, 0]
            elif data.ndim == 3:
                # 3D data
                slice_2d = data[:, :, slice_idx]
            else:
                print(f"Warning: Skipping {nifti_path.name}, unsupported dimension {data.ndim}")
                continue

            # --- Normalize and convert to 8-bit integer for saving as PNG ---
            # This step is crucial for converting float data to a standard image format
            slice_2d = slice_2d.astype(np.float32)
            if np.max(slice_2d) > np.min(slice_2d):
                slice_normalized = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d))
            else:
                slice_normalized = np.zeros_like(slice_2d) # Handle case of a black image
            
            slice_uint8 = (slice_normalized * 255).astype(np.uint8)

            # Create a PIL Image
            pil_img = Image.fromarray(slice_uint8)

            # Construct the output path
            # Example: 'subject_01.nii.gz' -> 'subject_01.png'
            output_filename = nifti_path.stem.replace('.nii', '') + '.png'
            output_path = output_dir / output_filename
            
            # Save the image
            pil_img.save(output_path)

        except Exception as e:
            print(f"Error processing {nifti_path.name}: {e}")

if __name__ == "__main__":
    # --- Configuration ---
    # Define the base directory where your source folders are located
    base_source_dir = Path("./") # Assuming the script is in the same dir as Train, Test, etc.
    
    # Define the main output directory
    base_output_dir = Path("./data")

    # --- Define source and destination paths ---
    source_dirs = {
        "train_images": base_source_dir / "Train",
        "train_labels": base_source_dir / "Train_label",
        "test_images":  base_source_dir / "Test",
        "test_labels":  base_source_dir / "Test_Label"
    }
    
    output_dirs = {
        "train_images": base_output_dir / "train/images",
        "train_labels": base_output_dir / "train/labels",
        "test_images":  base_output_dir / "test/images",
        "test_labels":  base_output_dir / "test/labels"
    }

    # --- Run the processing ---
    process_and_save_slices(source_dirs["train_images"], output_dirs["train_images"], "train image")
    process_and_save_slices(source_dirs["train_labels"], output_dirs["train_labels"], "train label")
    process_and_save_slices(source_dirs["test_images"], output_dirs["test_images"], "test image")
    process_and_save_slices(source_dirs["test_labels"], output_dirs["test_labels"], "test label")

    print("\nPreprocessing complete.")
    print(f"All 2D images saved in '{base_output_dir}'")