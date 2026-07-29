# cardiac-mri-diagnosis
A MRI Cardiac Diagnosis model, using patient data collected by academics across the world, using machine learning (PyTorch)

## Dataset description

It is a Time Series Dataset
contains 2075259 measurements gathered in a house located in Sceaux (7km out from Paris, France) between December 2006 and November 2010 (47 months).  
It has 9 features and does contain missing values (~1.25%)

![License](https://img.shields.io/github/license/olbailey/household-electricity-demand-forecasting)

## How to run

Developed using Python 3.12

### cd to the project main directory in your terminal

### For linux / Mac

1. First run `python3 -m venv .venv`
2. Activate using: `source .venv/bin/activate`
3. Finally `pip install -r requirements.txt`  
   **if you have an nvidia GPU otherwise:** requirements-torch-cpu.txt <br>
   **~Note:** Mac users may prefer to install torch separately via `pip install torch` for M-series optimization.

### Windows

Use the command prompt not powershell

1. First run `python -m venv .venv`
2. Activate using: `.venv/Scripts/activate`
3. Finally `pip install -r requirements.txt`  
   **if you have an nvidia GPU otherwise:** requirements-torch-cpu.txt

### <u>Next</u>
follow data/README.md for instructions for setting up the Dataset

### Install the package in editable mode
This is so your local source is used, which fixes issues with relative imports, and package-relative paths: <br>
```bash
pip install -e .
```
Run scripts as modules rather than as standalone files:

```bash
python -m package.script

e.g. python -m training.train
```

### Testing
To run tests for the programs, just run `pytest` in the terminal

## Data Attribution

> **Data Source:** [Automated Cardiac Diagnosis Challenge - MICCAI'17](https://www.kaggle.com/datasets/samdazel/automated-cardiac-diagnosis-challenge-miccai17)  
> **Author:** [Sayan Banerjee](https://www.kaggle.com/samdazel)  
> **Published:** November 2023  
> **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)  
> **DOI / Citation:** `10.34740/kaggle/dsv/7040769`

## <ins>Citation</ins>

O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al.
"Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved ?" in IEEE Transactions on Medical Imaging, vol. 37, no. 11, pp. 2514-2525, Nov. 2018
doi: 10.1109/TMI.2018.2837502