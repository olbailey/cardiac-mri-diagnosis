# Data

## Breakdown

### Patient Classes:
1. NOR (Normal): Normal cardiac anatomy and function with an ejection fraction > 50%, diastolic wall thickness < 12 mm, and specific volume criteria.
1. MINF (Systolic Heart Failure with Infarction): Ejection fraction < 40%, abnormal myocardial contractions, and potential LV remodelling.
1. DCM (Dilated Cardiomyopathy): Ejection fraction < 40%, LV volume > 100 mL/m2, and diastolic wall thickness < 12 mm.
1. HCM (Hypertrophic Cardiomyopathy): Normal cardiac function (ejection fraction > 55%) with myocardial segments thicker than 15 mm in diastole.
1. ARV (Abnormal Right Ventricle): Abnormal RV parameters with or without normal LV function.
Acquisition Protocol:

### Description
Data acquired over a 6-year period using 1.5 T and 3.0 T MRI scanners.<br>
Cine MR images obtained with SSFP sequence in breath-hold, with retrospective or prospective gating.<br>
Short-axis slices covering LV from base to apex, varying slice thickness (5-10 mm) and spatial resolution (1.34 to 1.68 mm2/pixel).<br>
28 to 40 volumes acquired per patient, introducing natural variability in image quality and coverage.<br>

Training and Testing Dataset:<br>
Data converted to NIfTI format without resolution loss.<br>
Expert-identified End-Diastolic (ED) and End-Systolic (ES) frames.<br>
Training set: 100 patients (20 per group) with manual references and group labels.<br>
Testing set: 50 patients (10 per group) with private manual references and group labels.<br>
Patient weight and height included for normalization of physiological parameters with body surface area (BSA).

# Setup

## Downloading Dataset
### Linux / Mac
Run `python3 scripts/download_data.py`
### Windows
Run `python .\scripts\download_data.py`

### If SSL is being blocked
1. Find the data downloadable straight from here: 
[Kaggle Cardiac database website](https://www.kaggle.com/datasets/samdazel/automated-cardiac-diagnosis-challenge-miccai17)
1. Download the data as a zip found as an option in the Download button at the top right of the website
1. Extract the zip folder
1. Copy the training and testing folders found at archive/database/ into the data/raw folder within this project
1. continue with the instructions for preprocessing the data

## Preprocessing Dataset
### Linux / Mac
Run `python3 scripts/preprocess_data_2d.py`
### Windows
Run `python .\scripts\preprocess_data_2d.py`