# Building Extraction from Remote Sensing Images using 3DJA-UNet3+

## Introduction

This repository contains the implementation of the 3DJA-UNet3+ model proposed in the paper "A method for extracting buildings from remote sensing images based on 3DJA-UNet3+". The model effectively segments buildings in remote sensing imagery by leveraging 3D Jump Distance Aggregation (3DJA) and an enhanced U-Net3+ architecture.

## Features

* **3DJA Module:** Captures multi-scale contextual information and enhances feature representation using 3D Jump Distance Aggregation.
* **U-Net3+ Architecture:** Employs a refined U-Net3+ architecture for efficient feature extraction and accurate segmentation.
* **End-to-End Trainable:** The model is designed for end-to-end training, seamlessly integrating the 3DJA module and U-Net3+ backbone.
* **High Accuracy:** Demonstrates superior performance in building extraction tasks compared to existing methods.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ibrandiay/3DJA-UNet3Plus.git

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
3. **Run the training script:**
   ```bash
   python train.py --data-path <path_to_dataset> --epochs <number_of_epochs>
   ```
   
4. **Run the evaluation script:**
   ```bash
   python evaluate.py --data-path <path_to_dataset>
   ```
   
5. **Run the inference script:**
   ```bash
   python inference.py --data-path <path_to_dataset>
   ```
   
## Contributing
Contributions are welcome! Submit pull requests or open issues.

## Citation
If you use this code, please cite the following paper:
Li, Y., Li, Y., Zhu, X. et al. A method for extracting buildings from remote sensing images based on 3DJA-UNet3+. Sci Rep 14, 19067 (2024). https://doi.org/10.1038/s41598-024-70019-z


