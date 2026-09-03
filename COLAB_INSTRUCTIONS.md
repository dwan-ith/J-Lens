# How to Run the Colab Notebook (Qwen3-8B on T4 GPU)

## Step 1: Open Google Colab
1. Go to https://colab.research.google.com
2. Click "File" → "Upload notebook"
3. Upload `colab_jlens_8b.ipynb` from `C:\Users\aacer\Desktop\MATS\`

## Step 2: Enable GPU
1. Click "Runtime" → "Change runtime type"
2. Under "Hardware accelerator", select **T4 GPU**
3. Click "Save"

## Step 3: Run All Cells
1. Click "Runtime" → "Run all"
2. Wait for the first cell to install dependencies (~2 min)
3. The model download will take ~5 min (8B = ~16GB)
4. The full experiment runs ~30 min on T4

## Step 4: Download Results
After the notebook finishes:
1. In the left sidebar, click the folder icon
2. Right-click `results/` folder → "Download"
3. You'll get the plots and data as a zip file

## What the notebook does
- Downloads Qwen3-8B and pre-fit J-Lens
- Runs the same experiments as our local run but on a larger model
- Tests whether the proximity artifact holds at 8B scale
- Tests whether the norm distortion occurs (it might not — larger models often have better-conditioned norms)
- Generates all figures

## If you get stuck
- **Out of memory**: T4 has 16GB VRAM. The notebook should fit, but if not, reduce `N = 3` to `N = 2` in the prompts cell.
- **Download fails**: The model is large. Try again — Colab sometimes has connection issues.
- **Pre-fit J-Lens unavailable**: If neuronpedia doesn't have Qwen3-8B J-Lens yet, the notebook will skip that section.
