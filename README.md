# Optimized-EEGNet-for-Consumer-Grade-EEG-Emotion-Recognition
optimized implementation of EEGNet specifically designed for consumer-grade EEG devices and emotion recognition tasks.
This repository contains the full implementation and analysis pipeline for my research study on robustness in EEG-based emotion recognition**, focusing  on noisy, consumer-grade EEG data like the DREAMER dataset.

## REQUIRMENTS

# Core Dependencies
numpy>=1.19.0
scipy>=1.7.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
matplotlib>=3.3.0
seaborn>=0.11.0

# Data Processing
pandas>=1.3.0

# Optional but Recommended
jupyter>=1.0.0
notebook>=6.4.0
ipykernel>=6.0.0

# Development Tools (Optional)
pytest>=6.0.0
black>=21.0.0
flake8>=4.0.0


# tensorflow-gpu>=2.8.0

# Specific versions for reproducibility 
# numpy==1.21.6
# scipy==1.7.3
# scikit-learn==1.1.1
# tensorflow==2.9.1
# matplotlib==3.5.2
# seaborn==0.11.2
# pandas==1.4.3
## System Requirements

- **Hardware**: MacBook Pro 13” (2017), Intel Core i5, 8GB RAM  
- **OS**: macOS Ventura 13.7.6
- **Python version**: `3.10+`

##  Acknowledgments

Original EEGNet paper: Lawhern et al. (2018)
@article{Lawhern2018,
  author={Vernon J Lawhern and Amelia J Solon and Nicholas R Waytowich and Stephen M Gordon and Chou P Hung and Brent J Lance},
  title={EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces},
  journal={Journal of Neural Engineering},
  volume={15},
  number={5},
  pages={056013},
  url={http://stacks.iop.org/1741-2552/15/i=5/a=056013},
  year={2018}
}
**Dataset: [DREAMER](https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions)
DREAMER dataset: Katsigiannis & Ramzan (2017)
TensorFlow team for the deep learning framework
Consumer EEG research community

## Project Overview

- 
- **Models Compared**:
  - EEGNet (8k parameters)
  - LSTM-Attention (150k+ parameters)
- **Tasks**:
  - Binary classification on valence and arousal
  - Subject-independent and subject-dependent evaluation
-
- ## Finding : EEGNet avoids catastrophic failure modes (bimodal accuracy collapse) seen in larger architectures, despite being much simpler and faster to train.




## .gitnore file
### macOS ###
# General
.DS_Store
.AppleDouble
.LSOverride

# Icon must end with two \r
Icon


# Thumbnails
._*

# Files that might appear in the root of a volume
.DocumentRevisions-V100
.fseventsd
.Spotlight-V100
.TemporaryItems
.Trashes
.VolumeIcon.icns
.com.apple.timemachine.donotpresent

# Directories potentially created on remote AFP share
.AppleDB
.AppleDesktop
Network Trash Folder
Temporary Items
.apdisk

### macOS Patch ###
# iCloud generated files
*.icloud

# End of https://www.toptal.com/developers/gitignore/api/macos
