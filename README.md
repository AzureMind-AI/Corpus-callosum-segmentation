# AURA-Net: Rapid and Accurate Corpus Callosum Segmentation

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-AURA--Net-green)]()

This project presents **AURA-Net**, a novel deep learning approach for segmenting the corpus callosum (CC) in brain MRI images. We developed a custom U-Net-based architecture that integrates Residual blocks, Attention gates, Atrous Spatial Pyramid Pooling (ASPP), and Deep Supervision. 

Our model significantly improves segmentation accuracy, outperforming existing state-of-the-art methods (Classic U-Net and U-Net++) on benchmark datasets and achieving the best reported results to date.

---

## 📢 Model Weights & Data
We provide the official pre-trained weights and example data for immediate testing.

### 🚀 **[Download Pre-trained Model Weights](https://drive.google.com/file/d/1dMhYHSGIT7l1wC8M6HRW6Iw9GuNYAWuE/view?usp=sharing)**
> **Note:** These weights were trained on 554 T1-weighted MRI scans using a nested 10-fold cross-validation scheme.

### 📂 **[Download Example MRI Data (.nii)](https://drive.google.com/file/d/1JxccJmtoodnsYEyIsHcVDSFvO8c6ySoh/view?usp=sharing)**
> Includes 4 example NIfTI files to test the inference pipeline.

---

## 📄 Abstract
The morphology of the midsagittal cross-section of the corpus callosum (CC) is sensitive to many neurological and psychiatric conditions, yet its small size and variability in shape complicate automatic segmentation on structural MRI. 

**AURA-Net** is an enhanced U-Net designed to produce reliable CC segmentations suitable for large neuroimaging studies. By combining multiscale context, channel-wise attention, residual learning, and deep supervision, AURA-Net delivers fast, accurate, and reproducible CC segmentations. This provides a practical foundation for studies utilizing callosal morphology as a biomarker for disease progression or treatment response.

## 🏗️ Architecture: AURA-Net
AURA-Net improves upon the standard U-Net framework by incorporating four key architectural enhancements to handle the specific challenges of CC segmentation (shape variability and proximity to the fornix):

1.  **Residual Blocks:** Replaces standard convolutional blocks to improve gradient flow and optimization stability.
2.  **Attention Gates:** Integrated into skip connections to suppress irrelevant background features and focus on the CC.
3.  **ASPP Module:** Located at the bottleneck to capture multi-scale contextual information.
4.  **Deep Supervision:** Auxiliary prediction heads to guide intermediate decoder layers during training.

## 📊 Performance & Results

The model was trained and evaluated on **554 T1-weighted MRI scans** (ADNI, Narratives, and Wahlheim cohorts). Images were pre-aligned to a common reference space (PIL orientation).

### Quantitative Metrics
AURA-Net was evaluated using a nested 10-fold cross-validation strategy.

| Metric | AURA-Net | Classic U-Net | U-Net++ |
| :--- | :---: | :---: | :---: |
| **Dice Similarity Coefficient (DSC)** | **98.82% (±0.32)** | 97.2% | 97.1% |
| **Accuracy** | **99.97%** | - | - |
| **Sensitivity** | **98.42%** | - | - |
| **Jaccard Similarity (JSC)** | **97.66%** | - | - |

> *AURA-Net significantly outperformed U-Net and U-Net++ (p ≈ 0.002, Wilcoxon signed-rank test).*

### Inference Speed
* **Speed:** < 1 second per subject on a single GPU.
* **Suitability:** Optimized for large-scale neuroimaging studies.

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.8+
* PyTorch
* NVIDIA GPU (Recommended for fast inference)

### Installation
```bash
git clone [https://github.com/AzureMind-AI/Corpus-callosum-segmentation.git](https://github.com/AzureMind-AI/Corpus-callosum-segmentation.git)
cd Corpus-callosum-segmentation
pip install -r requirements.txt
````

### Inference

To run segmentation on the provided example data:

1.  Download the [Example Data](https://drive.google.com/file/d/1JxccJmtoodnsYEyIsHcVDSFvO8c6ySoh/view?usp=sharing) and extract to `data/examples`.
2.  Download the [Model Weights](https://drive.google.com/file/d/1dMhYHSGIT7l1wC8M6HRW6Iw9GuNYAWuE/view?usp=sharing) and place in `weights/`.
3.  Run the inference script (example command):

<!-- end list -->

```bash
python inference.py --input_path data/examples --weights weights/aura_net_best.pth --output_path results/
```

## 📚 Datasets

The model was developed using data from:

1.  **ADNI (Alzheimer’s Disease Neuroimaging Initiative)**: 458 scans.
2.  **Narratives Database**: 55 scans.
3.  **Wahlheim et al. Database**: 41 scans.

## 📝 Citation

If you use this code or model in your research, please cite our paper:

```bibtex
@article{auranet2025,
  title={AURA-Net Enables Rapid and Accurate Corpus Callosum Segmentation for Large-Scale Neuroimaging Studies},
  author={Ali Eskandarian, Amir Sariaslani, Hamid Abrishami Moghaddam, Babak A. Ardekani, for the Alzheimer’s Disease Neuroimaging Initiative},
  journal={IEEE Transactions on Medical Imaging (TMI)},
  year={2025}
}
```

## 🤝 Acknowledgements

Data collection and sharing for this project was funded by the Alzheimer's Disease Neuroimaging Initiative (ADNI) (National Institutes of Health Grant U01 AG024904) and DOD ADNI.

-----

*Maintained by [AzureMind-AI](https://www.google.com/search?q=https://github.com/AzureMind-AI)*
