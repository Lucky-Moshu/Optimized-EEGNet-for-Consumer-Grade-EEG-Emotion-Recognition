# Optimized-EEGNet-for-Consumer-Grade-EEG-Emotion-Recognition
optimized implementation of EEGNet specifically designed for consumer-grade EEG devices and emotion recognition tasks.
# Created by https://www.toptal.com/developers/gitignore/api/macos
# Edit at https://www.toptal.com/developers/gitignore?templates=macos



This repository contains the full implementation and analysis pipeline for my research study on robustness in EEG-based emotion recognition**, focusing  on noisy, consumer-grade EEG data like the DREAMER dataset.

---

## Project Overview

- **Dataset: [DREAMER](https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions)
- **Models Compared**:
  - EEGNet (8k parameters)
  - LSTM-Attention (150k+ parameters)
- **Tasks**:
  - Binary classification on valence and arousal
  - Subject-independent and subject-dependent evaluation
- **Key Finding**: EEGNet avoids catastrophic failure modes (bimodal accuracy collapse) seen in larger architectures, despite being much simpler and faster to train.

---
## System Requirements

- **Hardware**: MacBook Pro 13” (2017), Intel Core i5, 8GB RAM  
- **OS**: macOS Ventura 13.7.6
- **Python version**: `3.10+`

---

@article{paredes2025less,
  title={When Less is More: How Architectural Simplicity Prevents Catastrophic Overfitting in Consumer-Grade EEG},
  author={Paredes Ocaranza, Carlos Rodrigo},
  year={2025},
  journal={Under Review}
}



**Paper Citation**
If you use the EEGNet model in your research and found it helpful, please cite the following paper:

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
