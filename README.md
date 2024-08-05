# Tumor Segmentation Algorithm
## Overview

This project involves developing and training a tumor segmentation algorithm for CT image patches. The dataset consists of 64x64x64 patches of CT images, each containing tumors in the middle. The primary objective is to accurately segment tumors in both clean and noisy image patches.

## Dataset

The dataset contains CT image patches and their corresponding segmentation masks. The images have been preprocessed to isotropic resolution, with HU values clipped at (-1024, 300) and rescaled to the range (0-255).

## Prerequisites 
Ensure you have the following installed: 
- Docker 
- NVIDIA Docker (for GPU support) 
- Make
- CUDA

## Directory Structure

The pipeline is organized as follows:

├── datasets

│   ├── __init__.py

│   ├── tumor_datasets.py

├── experiments

│   └── baseline.yaml

├── learning

│   ├── __init__.py

│   └── tumor_learning.py

├── pipeline.py

├── transforms

│   ├── __init__.py

│   └── transforms.py

└── utils.py


## Data preprocessing
To prepare the input data for the model, a CSV file is generated using the `to_folds.ipynb` notebook. This CSV file contains pairs of `cts_path` and `masks_path`, where each row represents a CT scan and its corresponding segmentation mask.

The `cts_path` corresponds to the file path of the CT scan, while the `masks_path` corresponds to the file path of the segmentation mask. These paths point to the respective CT scan and mask files stored in the dataset directories.

The notebook iterates through the CT scan files (`*.nii.gz`) in the `cts_dir` directory and matches them with their corresponding mask files in the `masks_dir` directory based on their filenames. It then creates a DataFrame with columns `cts_path`, `masks_path`, and `fold`, where `fold` represents the fold to which the data point belongs (e.g., train, validation, or test).

After splitting the data into train, validation, and test sets, the DataFrame is saved as CSV files for further use by the segmentation pipeline. These CSV files are used to provide input data to the segmentation algorithm during training and evaluation.

## Configuration

The `experiments/baseline.yaml` file contains configuration details such as paths to the training and testing data, model parameters, optimizer settings, etc.



## Project Files

### `pipeline.py`

This is the main script to run the experiment. It handles loading the configuration, setting up the dataset and data loaders, initializing the model, criterion, optimizer, and scheduler, and managing the training and evaluation process.

### `Dockerfile`

A Dockerfile to create a Docker image with all the dependencies required to run the project.

### `Makefile`

A Makefile to provide convenient commands to build and run the Docker container.

### `experiments/baseline.yaml`

Configuration file for the experiment. Modify this file to adjust parameters such as dataset paths, model settings, optimizer settings, etc.

### `datasets/tumor_datasets.py`

Defines the dataset class for loading and processing the CT image patches and their corresponding segmentation masks.

### `transforms/transforms.py`

Defines the transformations and augmentations applied to the data during training and validation.

### `learning/tumor_learning.py`

Defines the learning process, including training and validation steps, loss calculation, and metric evaluation.

### `utils.py`

Contains utility functions for tasks such as loading configuration files and setting random seeds.

## Running the Project

### Using Makefile

1.  **Install Dependencies**: Execute the following command to install the necessary dependencies:
```
make install
```
2. **Run the Pipeline**: Execute the following command to run the pipeline with the configuration file:
```
make run
```
3. **Development**: If you are working on the project, update the dependencies and run the pipeline using the following commands:
```
make run-dev
```

### Without Makefile

1.  **Install the dependencies**: Ensure you have Python 3.9+ and all the required packages installed. You can install the required packages using:
```
pip install -r requirements.txt
```

2. **Run the pipeline**: Execute the pipeline script with the configuration file:
```
python pipeline/pipeline.py pipeline/experiments/baseline.yaml
```


## Checkpointing

The best model weights, based on the `mean_dice_val` metric, are automatically saved during training. These weights are saved in the directory (You can change the directory in the configuration file `experiments/baseline.yaml`):

```
/home/yerkyn/tumor_seg/logs/unet_test_tumor/
```

Ensure this directory exists and has the appropriate permissions for saving the checkpoint files.


## Author

This segmentation project was developed by Yerkyn Yesbay.
Email: yesbay185@gmail.com
