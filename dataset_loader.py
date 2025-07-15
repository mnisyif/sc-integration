# # ── preview_swin_dataset.py ─────────────────────────────────────────────────────
# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import ast
import io
from PIL import Image
from typing import Optional
import ast
import io
from PIL import Image

# # %%
# # ---------------------------------------------------------------------
# # 1.  Point to the directory that holds the three .npy files
# #     (edit snr, model, or folder names as needed)
# # ---------------------------------------------------------------------
# snr      = 10
# model    = "swin"
# base_dir = Path("GenSC-Testbed") / "AWGN_Generated_Classification" / "Train" / str(snr)

# features  = np.load(base_dir / f"{model}_features.npy")
# labels    = np.load(base_dir / f"{model}_labels.npy")
# filenames = np.load(base_dir / f"{model}_filenames.npy", allow_pickle=True).astype(str)

# # quick sanity‑check
# assert len(features) == len(labels) == len(filenames), "arrays mis‑aligned"
# print(f"Loaded {len(labels):,} samples:", features.shape, features.dtype)

# # %%
# # ---------------------------------------------------------------------
# # 2.  Peek at a few rows in a DataFrame
# # ---------------------------------------------------------------------
# df = pd.DataFrame({
#     "filename": filenames,
#     "label"   : labels,
#     # don't shove all 1000 feature dims in the table—just show the first 5
#     **{f"f{i}": features[:, i] for i in range(5)}
# })
# display(df.head())        # Jupyter / VS Code; or print(df.head(10)) in plain REPL

# # %%
# # ---------------------------------------------------------------------
# # 3.  Show label distribution
# # ---------------------------------------------------------------------
# counts = pd.Series(labels).value_counts().sort_index()

# plt.figure(figsize=(8, 4))
# counts.plot(kind='bar')
# plt.title("Label distribution")
# plt.xlabel("Label ID")
# plt.ylabel("# samples")
# plt.tight_layout()
# plt.show()

# # %%
# # ---------------------------------------------------------------------
# # 4.  (Optional) quick PCA → 2‑D scatter to see structure
# # ---------------------------------------------------------------------
# try:
#     from sklearn.decomposition import PCA
#     pca = PCA(n_components=2).fit_transform(features.astype("float32"))
#     plt.figure(figsize=(6, 5))
#     plt.scatter(pca[:, 0], pca[:, 1], s=4, c=labels, alpha=0.5)
#     plt.title("First two PCA components")
#     plt.xlabel("PC‑1"); plt.ylabel("PC‑2")
#     plt.tight_layout()
#     plt.show()
# except ImportError:
#     print("sklearn not installed; skipping PCA plot")
# %%
data_dir = Path("GenSC-Testbed") / "data"

parquet_data = pd.read_parquet(data_dir / "test-00000-of-00003.parquet")

# Display the first few rows of the DataFrame
print(parquet_data.head())
parquet_data.info()
print(type(parquet_data))
# %%

def df_columns_to_csv(df: pd.DataFrame, out_dir: str = "csv_columns",
                      include_index: bool = False, float_format=None,
                      compression: str | dict | None = None):
    """
    Save each column in `df` to its own CSV file.

    Parameters
    ----------
    df : pd.DataFrame
    out_dir : str
        Folder to hold the CSVs (created if missing).
    include_index : bool
        Whether to write the DataFrame’s index as the first column.
    float_format : str | None
        e.g. '%.6f' to control numeric formatting.
    compression : str | dict | None
        Pass any value accepted by `to_csv(compression=...)`, e.g. 'gzip'.
    """
    os.makedirs(out_dir, exist_ok=True)
    for name in df.columns:
        df[[name]].to_csv(os.path.join(out_dir, f"{name}.csv"), index=include_index, header=True, float_format=float_format, compression=compression)
 # %%
df_columns_to_csv(parquet_data, out_dir="csv_cols", float_format="%.6g")  
# %%
def extract_images_from_parquet(df: pd.DataFrame, image_column: str = "image", 
                               output_dir: str = "extracted_images", 
                               filepath_column: str = "image_path",
                               preserve_structure: bool = True):
    """
    Extract image data from a parquet DataFrame and save as PNG files.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing image data
    image_column : str
        Name of the column containing image data
    output_dir : str
        Base directory to save extracted images
    filepath_column : str
        Column name containing the original file paths
    preserve_structure : bool
        Whether to preserve the original directory structure
    """
    success_count = 0
    error_count = 0
    
    for idx, row in df.iterrows():
        try:
            # Parse the image data (it's stored as a string representation of a dict)
            image_data_str = row[image_column]
            
            # Convert string to dict
            if isinstance(image_data_str, str):
                image_data = ast.literal_eval(image_data_str)
            else:
                image_data = image_data_str
            
            # Extract bytes data
            image_bytes = image_data['bytes']
            
            # Ensure we have proper bytes data
            if not isinstance(image_bytes, bytes):
                # Try to convert if it's a string or other type
                if isinstance(image_bytes, str):
                    image_bytes = image_bytes.encode('latin-1')
                else:
                    # If it's a numpy array or other type, convert to bytes
                    image_bytes = bytes(image_bytes)
            
            # Create PIL Image from bytes
            image = Image.open(io.BytesIO(image_bytes))
            
            # Generate output path based on original filepath
            if preserve_structure and filepath_column in df.columns:
                original_path = str(row[filepath_column])
                # Create the full output path preserving directory structure
                output_path = os.path.join(output_dir, original_path)
                
                # Create directories if they don't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            else:
                # Fallback to simple filename
                if filepath_column in df.columns:
                    # Extract just the filename from the path
                    filename = os.path.basename(str(row[filepath_column]))
                    output_path = os.path.join(output_dir, filename)
                else:
                    filename = f"image_{idx:06d}.png"
                    output_path = os.path.join(output_dir, filename)
                
                # Ensure output directory exists
                os.makedirs(output_dir, exist_ok=True)
            
            # Save image
            image.save(output_path)
            success_count += 1
            
            if success_count <= 10 or success_count % 100 == 0:
                print(f"Saved: {output_path}")
            
        except Exception as e:
            error_count += 1
            print(f"Error processing row {idx}: {e}")
            continue
    
    print(f"\nImage extraction completed!")
    print(f"Successfully saved: {success_count} images")
    print(f"Errors: {error_count}")
    print(f"Images saved to: {output_dir}")

# %%
# Extract images from the parquet data preserving directory structure
extract_images_from_parquet(parquet_data, 
                           image_column="image",
                           output_dir="extracted_images",
                           filepath_column="image_path",
                           preserve_structure=True)
# %%

# Alternative: Extract images without preserving structure (flat directory)
# extract_images_from_parquet(parquet_data, 
#                            image_column="image",
#                            output_dir="extracted_images_flat",
#                            filepath_column="image_path",
#                            preserve_structure=False)
# %%
