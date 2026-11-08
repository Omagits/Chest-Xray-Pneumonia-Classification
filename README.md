# Chest X-ray Pneumonia Classification with PyTorch

An educational computer-vision project using **pretrained ResNet-18** for binary classification of chest X-rays into **NORMAL** and **PNEUMONIA**, developed during DataCamp's *Classifying X-Ray Images using PyTorch* project and extended with validation and error analysis. DataCamp marked the exercise **Correct solution**.

## Approach

- ImageNet-pretrained ResNet-18 with its backbone weights frozen; replace the final fully connected layer with a single logit.
- ImageNet normalization and binary cross-entropy with logits.
- Balanced DataCamp-provided subset: 300 training images (150 per class), 100 test images (50 per class).
- Stratified training/validation split: 240 training, 60 validation; batch size 16.
- Final experiment: Adam, learning rate `0.0003`, three epochs; frozen backbone in evaluation mode to preserve BatchNorm statistics.
- Sigmoid threshold of 0.5 for binary predictions.

## Recorded results (one experiment)

| Metric | Validation (60 images) | Test (100 images) |
|---|---:|---:|
| Accuracy | **90.0%** | **80.0%** |
| Pneumonia F1 | 89.3% | 82.1% |
| Pneumonia recall | 83.3% | 92.0% |
| Pneumonia precision | 96.2% | 74.2% |

Validation confusion matrix (rows = actual NORMAL/PNEUMONIA, columns = predicted NORMAL/PNEUMONIA): `[[29, 1], [5, 25]]`.

Test confusion matrix: `[[34, 16], [4, 46]]`. The test set contains 16 false pneumonia alerts and 4 missed pneumonia cases.

Training and validation losses over three epochs:

| Epoch | Train loss | Validation loss |
|---|---:|---:|
| 1 | 0.6415 | 0.5809 |
| 2 | 0.5697 | 0.5170 |
| 3 | 0.5107 | 0.4689 |

## Running the notebook

1. Create a Python environment and install `pip install -r requirements.txt`.
2. Provide your own appropriately licensed image subset in this structure:
   ```text
   data/chestxrays/
     train/NORMAL/
     train/PNEUMONIA/
     test/NORMAL/
     test/PNEUMONIA/
   ```
3. Open `notebook.ipynb` in Jupyter or VS Code and run top to bottom. The ImageNet pretrained weights may download on first run.
4. Outputs depend on library versions, randomness, and exact images; numbers above describe the recorded DataCamp run, not a guaranteed reproduction.

## Files and dataset usage

`notebook.ipynb` contains a streamlined runnable version of the final training/validation/test workflow. The original exported notebook and X-ray dataset are not republished. `data/` and model `.pth` files are excluded from Git by `.gitignore`. The original workspace contains a ~45 MB checkpoint; store it locally or publish separately with suitable model-file hosting and a reproducibility note.

## Limitations and ethics

This is a small, curated educational dataset, **not** a clinically validated diagnostic system. The same test set was used in earlier exploratory iterations, so the reported 80% test score **must not be presented as a fully independent, unbiased final benchmark**. Patient-level independence and external generalization have not been verified. False negatives and false positives make clinical use inappropriate.
