# EGA-Ploc: Efficient Global-Local Attention Model for Multi-label Protein Subcellular Localization

[![License](https://img.shields.io/badge/License-Academic%20Free%20License%20v3.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.11.0-red.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20JBHI-green.svg)](https://ieeexplore.ieee.org/document/11175487)

## 📖 Overview

**EGA-Ploc** is an advanced deep learning framework for multi-label protein subcellular localization prediction from immunohistochemistry (IHC) images. This repository contains the official implementation of our paper:

> **[EGA-Ploc: An Efficient Global-Local Attention Model for Multi-label Protein Subcellular Localization Prediction on the Immunohistochemistry Images](https://ieeexplore.ieee.org/document/11175487)**  
> *Accepted by IEEE Journal of Biomedical and Health Informatics, September 2025*

### 🎯 Key Features

- **🔬 Novel Linear Attention Mechanism**: Efficiently captures discriminative representations from IHC images
- **🏗️ Hierarchical Multi-scale Architecture**: Preserves both fine-grained subcellular patterns and global spatial context
- **⚖️ Enhanced Multi-label Objective**: Counteracts dataset imbalance through joint optimization
- **🚀 High Performance**: Outperforms existing patch-based and downsampling-reliant approaches

## 🚀 Quick Start

### Online Demo

Experience EGA-Ploc instantly through our Hugging Face Space:  
👉 **[Live Demo](https://huggingface.co/spaces/austinx25/EGA-Ploc)**

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/gd-hxy/EGA-Ploc.git
   cd EGA-Ploc
   ```

2. **Set up environment**
   ```bash
   conda env create -f environment.yaml
   conda activate Vislocas
   ```

3. **Download datasets**
   - **Vislocas Dataset**: [Zenodo](https://doi.org/10.5281/zenodo.10632698)
   - **HPA18 Dataset**: [GraphLoc](http://www.csbio.sjtu.edu.cn/bioinf/GraphLoc) or contact `kooyang@aliyun.com`

4. **Download pre-trained models** (optional)
   - **Vislocas Model**: [Download Link](https://jxstnueducn-my.sharepoint.com/:f:/g/personal/wanboyang_jxstnu_edu_cn/EpEDB3GcXMZFvRz9lQaBHswBYTEWUDF6ThPBHWqEPB-eUQ?e=jsSoY0)
   - **HPA18 Model**: Same repository as above

## 🏗️ Model Architecture

EGA-Ploc integrates three key components:

1. **Efficient Global-Local Attention**: Novel linear attention mechanism for computational efficiency
2. **Multi-scale Feature Fusion**: Hierarchical architecture capturing both local and global patterns
3. **Balanced Multi-label Learning**: Enhanced objective function addressing class imbalance

### Model Variants

- **ETP_cls_l1/l2/l3**: Base model variants with different backbone complexities
- **ETP_cls_cl0/cl1/cl2/cl3**: Cascaded backbone variants
- **Feature Fusion Models**: Multi-scale feature integration variants (featureAdd234, featureAdd324, etc.)

## 📊 Performance

EGA-Ploc achieves state-of-the-art performance on both Vislocas and HPA18 datasets, demonstrating superior accuracy in multi-label protein subcellular localization tasks.

## 📁 Project Structure

```
EGA-Ploc/
├── assets/                 # Sample images for demo testing
├── data/                   # Dataset annotation files (.csv)
├── datasets/               # Data loading modules
│   ├── ihc.py             # Vislocas dataset loader
│   ├── HPA18.py           # HPA18 dataset loader
│   ├── build.py           # Dataset builder
│   └── loader.py          # Data loader utilities
├── models/                 # Core model implementations
│   └── ETPLoc/            # EGA-Ploc model architecture
│       ├── backbone.py    # Backbone networks
│       ├── cls.py         # Classification heads
│       ├── nn/            # Neural network modules
│       └── utils/         # Model utilities
├── tools/                  # Training and testing scripts
│   ├── test.py            # Full dataset evaluation
│   ├── test_demo.py       # Single image testing
│   └── train.py           # Model training
├── utils/                  # Utility functions
│   ├── args.py            # Command-line arguments
│   ├── checkpoint.py      # Model checkpointing
│   ├── eval_metrics.py    # Performance evaluation
│   └── optimizer.py       # Optimization algorithms
└── results/               # Output directory for models and predictions
```

## 🛠️ Usage

### Testing on Full Datasets

```bash
# Test on Vislocas dataset
python tools/test.py --dataset IHC

# Test on HPA18 dataset  
python tools/test.py --dataset HPA18
```

### Single Image Testing

```bash
# Test a single image
python tools/test_demo.py --dataset IHC --single_image_path ./assets/Cytopl;Mito/55449_A_1_2.jpg
```

### Training

Training code will be released in mid-October 2025.

## 📋 Requirements

- **Platform**: Ubuntu 20.04+ (Windows/Linux supported)
- **GPU**: NVIDIA GPU with CUDA 11.3+
- **Python**: 3.8.15
- **PyTorch**: 1.11.0
- **Dependencies**: See `environment.yaml` for complete list

## 📄 License

### Academic Use
This project is released under the **Academic Free License v3.0** for non-commercial research and educational purposes. You are free to:

- ✅ Use, copy, and modify for academic research
- ✅ Share and distribute for educational purposes
- ✅ Build upon the work for non-commercial applications

### Commercial Use
For commercial licensing, please contact the authors to discuss terms and conditions. Commercial use requires explicit permission and may be subject to licensing fees.

## 🤝 Citation

If you use EGA-Ploc in your research, please cite our paper:

```bibtex
@article{egaploc2025,
  title={EGA-Ploc: An Efficient Global-Local Attention Model for Multi-label Protein Subcellular Localization Prediction on the Immunohistochemistry Images},
  author={Wan, Boyang and others},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2025},
  doi={10.1109/JBHI.2024.11175487}
}
```

## 📞 Contact

For questions, issues, or collaboration opportunities:

- **Email**: `kooyang@aliyun.com`
- **Issues**: [GitHub Issues](https://github.com/gd-hxy/EGA-Ploc/issues)

## 🙏 Acknowledgments

We thank the contributors and the research community for their valuable feedback and support in developing EGA-Ploc.

---

*EGA-Ploc: Advancing protein localization prediction through efficient deep learning architectures.*
