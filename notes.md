I've read through the repo. Here's the overview, then suggestions in priority order.

## Overview

The project segments cardiac MRI from the ACDC dataset, and the plan is to add diagnosis classification later.

**Pipeline:**
1. **`scripts/download_data.py`** downloads the ACDC data from Kaggle into `data/raw`, and falls back to manual download instructions if SSL fails.
2. **`scripts/preprocess_data_2d.py`**, for each patient:
   - takes the first labelled frame;
   - resamples it in-plane to 1.5 mm;
   - clips intensities to the 0.5–99.5 percentiles and z-scores them;
   - saves each 2D slice as `.npy`, with the diagnosis in the filename (`patient120_slice02_MINF.npy`).
3. **`src/dataset/mri_dataset.py`** is a MONAI-based Dataset that pads or crops slices to 256×256.
4. **`src/training/segmentation.py`** trains an `smp.Unet` (pretrained ResNet34) with DiceCE loss and reports Dice for RV, myocardium and LV.
5. **`src/model/u_net.py`** is your own U-Net with configurable depth. It's tidy, but it isn't used at the moment.
6. **`src/data/`** holds exploration tools: the overlay viewer and dataset stats.
7. **`classification.py`** is empty so far.

The structure is sensible (a `src/` layout, a config file, separate preprocessing) and the READMEs on the dataset are good.

## Bugs and correctness problems (fix these first)

2. **The test set is used for validation.** `evaluate` runs on `data/processed/testing` every epoch. Once you start early stopping or tuning on that, your test score will look better than it really is. Hold out about 15–20% of the training patients as a validation set, splitting by patient rather than by slice so slices from one patient can't leak across, and only touch the test set at the end.
5. **Models are only saved, and the prediction plot only shown, on Ctrl+C.** If training finishes normally, the result is lost. Move that code after the loop, and save the best checkpoint by validation Dice automatically.
6. **The config is mostly ignored.** `unet.yaml` says `encoder_depth: 4` and `base_channels: 64`, but the model is hardcoded as `resnet34`, depth 5, 4 classes. Either drive the model choice from the config (e.g. `type: smp_unet | custom_unet`) or remove the unused keys.
7. **The training script runs on import.** All of `segmentation.py` sits at module level. Wrap it in `main()` behind `if __name__ == "__main__":`.

## Data and modelling

- **Use both labelled frames (ED and ES).** You currently take only the lowest frame, which is end-diastole (ED). End-systole (ES) doubles your training data. More importantly, the ED/ES volume difference gives the ejection fraction, which is the main feature separating NOR, DCM, MINF and HCM. Without ES the classification stage has little to work with.
- **Plan the classification stage around features, not images.** With 100 training patients, a CNN classifier on raw images will overfit. The approach that works on ACDC is to:
  - segment ED and ES;
  - compute LV/RV volumes, EF, myocardial mass and wall thickness, normalised by body surface area (height and weight are in `Info.cfg`);
  - train a small classifier such as a random forest or gradient boosting on those features.

  Published results reach about 90%+ accuracy this way.
- **Turn augmentation back on.** It's commented out. Mild affine transforms, noise and contrast/gamma changes help a lot on a dataset this small.
- **Report metrics the way the ACDC challenge does.** Slice-averaged 2D Dice flatters the model: empty apex and base slices are easy. Rebuild each patient's 3D volume and report per-patient 3D Dice plus Hausdorff distance, with ED and ES reported separately.
- **Sort `os.listdir`** in `MriDataset` and fix the random seeds, so runs can be reproduced.
- Later on: compare against nnU-Net on ACDC to see how close you are, and try 2.5D input (neighbouring slices as extra channels).

## Code quality and setup
- **Package names are generic.** Top-level packages called `utils`, `data`, `model` and `dataset` can clash with other installed packages. Put them under one package, e.g. `src/cardiac_mri/...`.
- **The README is out of date:**
  - The example `python -m training.train` no longer exists; it's now `training.segmentation`.
  - The scripts import from `utils`, so `pip install -e .` has to happen *before* the data steps. The README currently sends you to the data instructions first.
- **There are no tests,** although the README says to run `pytest`. Good first ones:
  - `resize_image` keeps the physical size;
  - `get_patient_frame` and `get_patient_label`;
  - the U-Net output shape for a few depths;
  - the dataset returns `(1, 256, 256)` tensors of the right dtypes.
- **Small cleanups:**
  - **Duplicated constants:** `LABEL_NAMES`/`LABEL_COLORS` are defined twice, and so is the `process_patient` logic (in `stats.py` and in preprocessing).
  - **`MriDataset` method names:** the method `retreive_file` is misspelled and unused, and `apply_transform` repeats `__getitem__`.
  - **`image-mask-analysis.py`:** the hyphen means it can't be imported, and its docstring calls it `view_nii_overlay.py`.
  - **`get_patient_frame`:** it's typed as returning `int` but returns a string.
  - **Dead code:** there's a lot of commented-out code and debug lines.
- **Nice to have:** log metrics per epoch (TensorBoard or MLflow; you already have an mlflow placeholder), save the config next to each checkpoint, and use mixed precision on GPU.

If you only do three things now: fix the scheduler, add a patient-level validation split, and preprocess both ED and ES frames. I'm happy to make any of these changes.
