# 


# NFSR
This is a Python implementation of `Similarity Regression of Functions in Different Compiled Forms with Neural Attentions on Dual Control-Flow Graphs` .

## Introduction
This project is a Python implementation of a research paper focused on binary similarity detection methods. The objective of the paper is to develop algorithms and techniques for effectively identifying similarities between binary data.

This implementation aims to make the methods proposed in the paper more accessible and usable by the community. It replicates the key algorithms and approaches discussed in the paper, providing a hands-on tool for researchers and professionals interested in the field of binary similarity detection.

## Dependencies

- Python >= 3.7
- angr >= 9.2.6
- angr-utils >= 0.5.0
- numpy >= 1.21.6
- matplotlib >= 3.5.2
- tensorflow >= 2.4.1
- torch >= 1.8.0

## Installation
Describe how to install the project and all dependencies.
```bash
git clone https://github.com/your-username/project-name.git
cd project-name
pip install -r requirements.txt
```

## Usage
This project consists of two main steps: preprocessing the data using feature generation, and then training the model using NFSR (some abbreviation or full form, if applicable). Below are the details on how to run each step:

### Dataset
please download form https://drive.google.com/drive/folders/1FXlrGiZkch9bnAxlrm43IhYGC3r5NveA

### Preprocessing

The preprocessing step involves generating features from the binary data. You can use the `feature_generator.py` script located in the `./src/preprocess/` directory for this purpose.

```bash
cd src/preprocess
python feature_generator.py
```

### NFSR Training

Once preprocessing is complete, you can proceed to train the model using the NFSR algorithm. Use the `train.py` script located in the ./src/ directory for this step.

```bash
cd ../  # go back to the root directory
cd src
python train.py
```
Make sure you have navigated to the appropriate directories before running these commands. Modify any parameters in the scripts as needed to suit your specific requirements.

## Contributing
Contributions are welcome in any form! Here are some ways you can contribute:

1. Raising issues or requesting features
2. Submitting changes

