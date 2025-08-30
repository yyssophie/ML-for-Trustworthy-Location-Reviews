# ML for Trustworthy Location Reviews — Colab Guide



This README explains how to set up **Google Colab**, run `test_colab_final.ipynb`, and reproduce the training/evaluation results.

---

## Quickstart (TL;DR)

1. Open `test_colab_final.ipynb` in Colab.
2. In Colab: **Runtime → Change runtime type → Hardware accelerator → GPU → Save**.
3. Run all cells **from top to bottom**.
4. Ensure the dataset file exists at: `data/out/augmented_shuffled.csv`.

---

## Requirements & Environment

- Google account + Google Colab
- Internet access (to clone the repo and install packages)
- GPU runtime (Colab **T4** is fine)
- Dataset CSV at `data/out/augmented_shuffled.csv`

The first bootstrap cell in the notebook will:

- Clone this repo on branch `main`
- Install **CUDA 12.4** compatible PyTorch wheels (`torch`, `torchvision`, `torchaudio`)
- Install Python dependencies from `requirements.txt`
- Verify GPU availability and print device info

If your data/results live in Google Drive, run the **Drive mount** cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## Dataset Schema (minimum columns)

- `business_name` (str)
- `description` (str or NaN)
- `category` (str / list-like as string)
- `text` (raw review text)
- `predicted_label` (one of: `Valid`, `Advertisement`, `Irrelevant`, `Rant_Without_Visit`)

The notebook maps `predicted_label → label` as:

- `Valid: 0`, `Advertisement: 1`, `Irrelevant: 2`, `Rant_Without_Visit: 3`

---

## What the Notebook Does

**Data prep & splits**

- Loads `data/out/augmented_shuffled.csv`
- Drops empty/NaN `predicted_label`
- **Sampling logic (kept exactly as designed):**
  - Randomly sample **300** rows from class **Valid** → add to a **pool**
  - The pool = **300 Valid** + **all non-Valid** rows
  - **80/20 stratified split** on the pool → (train\_pool, base\_test)
  - **Final test set** = `base_test` **+ all remaining Valid** rows that were **not** sampled into the pool
  - Build a **small validation set (< 50 samples)** from `train_pool` (defaults to \~5% of train\_pool, capped at 48)
  - The remainder of `train_pool` becomes the **training set**

**Modeling & training**

- Converts each row to a compact JSON-like string for classification
- Tokenizer/model: `nlptown/bert-base-multilingual-uncased-sentiment`
- Training: Hugging Face `Trainer`, **8 epochs**, FP16, weight decay, warmup, `f1_macro` used for model selection

**Evaluation & artifacts**

- Evaluation is run on the **test** split (large holdout)
- Prints classification report & confusion matrix
- **Exports** test data with labels to: `data/out/test_set_text_label.csv`

---

## How to Run (Step-by-Step)

1. Open the notebook in Google Colab.
2. Switch runtime to GPU.
3. Run the first cell (bootstrap). It will:
   - Clone the repo on `main`
   - Install CUDA 12.4 wheels for PyTorch
   - Install `requirements.txt`
   - Print GPU and Torch versions
4. (Optional) Mount Drive if you keep data there.
5. Run subsequent cells in order without modifications, unless you want to:
   - **Fix the seed** for reproducibility (`seed = 42`)
   - **Change the model** (edit the `checkpoint` string)
6. After training completes, scroll to the **evaluation** cell to see metrics.

---

## Expected Console Outputs

- **GPU & Torch check** (example):
  ```
  Torch: 2.6.0+cu124 | Built with CUDA: 12.4 | cuda.is_available: True
  GPU is available. Device name: Tesla T4
  ```
- **Split sizes** (values vary by seed, shape similar):
  ```
  Train pool size (pre-val): 869
  Small validation size (<50): 43
  Train size: 826
  Test size: 7599
  ```
- **DatasetDict** summary:
  ```
  train: 826 | validation: 43 | test: 7599
  ```

---

## Evaluating a Specific Checkpoint

You can evaluate any saved checkpoint after training.

**Find available checkpoints**:

```bash
!find results -type d -name "checkpoint-*"
```

**Point to one and run evaluation** (edit the path to match your run):

```python
checkpoint_dir = "results/nlptown/bert-base-multilingual-uncased-sentiment-952pm/checkpoint-3339"
```

The notebook cell will reload that checkpoint and print a classification report and confusion matrix. It evaluates on the **validation** split by default; you can change it to the **test** split if desired.

---

## Outputs Overview

- **Model checkpoints**: `results/<model-name>/checkpoint-*/`

- **(Optional)** Misclassifications helper: there’s a cell demonstrating how to create a misclassified samples dataframe&#x20;

---

## Troubleshooting

- **PyTorch installed without CUDA**: re-run the first bootstrap cell; it forces CUDA 12.4 wheels and re-validates.
- **Only one GPU in Colab**: the `CUDA_VISIBLE_DEVICES="1,2,3"` line has no effect on single-GPU Colab; safe to ignore.
- **Stratification errors**: ensure each class present in the pool has at least one sample (the provided dataset counts meet this condition).
- **OOM (out of memory)**: reduce `per_device_train_batch_size` (e.g., 8) and/or `max_length` (currently 256) in the tokenization function.

---

## Repo Structure (key paths)

```
ML-for-Trustworthy-Location-Reviews/
├─ data/
│  └─ out/
│     └─ augmented_shuffled.csv        # input (you provide)
│    
├─ results/
│  └─ <model-name>/checkpoint-XXXX/    # saved checkpoints
├─ test_colab_final.ipynb              # main notebook
├─ requirements.txt
└─ README.md
```

---

## Reproducibility Notes

The notebook currently sets `seed = random.randint(0, 100)`. For stable, repeatable splits and results, replace with a constant, e.g.:

```python
seed = 42
```

---

