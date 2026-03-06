# Hybrid DDoS Detection with Statistical and Semantic Features

## Overview

This notebook (`Final.ipynb`) implements a hybrid DDoS detection framework that combines:

- **PCA-reduced tabular (statistical) features** extracted from flow-level network data.
- **Semantic embeddings** derived from a pre-trained DDoSBert model (`Thi-Thu-Huong/DDoSBert`) using HuggingFace Transformers.
- **Logistic Regression** as a common classifier across all variants.

Four model variants are evaluated on two benchmark datasets (CICDDoS2019 and AdDDoSDN):

1. **(A) Baseline: PCA + LR**  
   Logistic Regression trained on PCA-reduced tabular features (with SMOTE on the training split).

2. **(B) SLM-only: Embeddings + LR (SMOTE)**  
   Logistic Regression trained solely on DDoSBert semantic embeddings, with SMOTE applied to training embeddings.

3. **(C) Hybrid: Attention Fusion → LR (SMOTE)**  
   Attention-based fusion of PCA features and semantic embeddings, followed by Logistic Regression on fused features (with SMOTE).

4. **(D) Hybrid: Concatenation → LR (SMOTE)**  
   Simple concatenation fusion of PCA features and semantic embeddings, followed by Logistic Regression on fused features (with SMOTE).

The notebook reports accuracy, macro-averaged precision, recall, and F1-score for each variant, along with confusion matrices and 5-fold cross-validation results (for both Attention Fusion and Concatenation variants).

---

## Environment and Dependencies

The notebook is designed to run in a Python 3 environment with GPU support (e.g., Google Colab with a Tesla T4 GPU). It installs additional libraries directly in the first cell.

Core dependencies:

- Python 3.x
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `imbalanced-learn`
- `torch` (PyTorch)
- `transformers`
- `accelerate`
- `huggingface_hub` (implicit via `transformers`)

To reproduce the environment outside Colab, install:

```bash
pip install numpy pandas scikit-learn matplotlib imbalanced-learn torch \
            transformers accelerate
```

> Note: For higher download limits from HuggingFace, configure `HF_TOKEN` in your environment, but the notebook will also run with anonymous access (subject to rate limits).

---

## Data Requirements

The notebook expects two CSV files (can be adjusted):

- **CICDDoS2019** (e.g., `cicddos2019.csv`)
- **AdDDoSDN** (e.g., `adddosdn_cicflow_dataset.csv`)

The paths are configured in the `DATASETS` dictionary:

```python
DATASETS = {
    "CICDDoS2019": "/content/cicddos2019.csv",
    "AdDDoSDN"   : "/content/adddosdn_cicflow_dataset.csv",
}
```

To run locally:

1. Place the CSV files somewhere on disk (e.g., `D:/k20proj/data/...`).
2. Update `DATASETS` to point to your local paths, for example:

```python
DATASETS = {
    "CICDDoS2019": "D:/k20proj/data/cicddos2019.csv",
    "AdDDoSDN"   : "D:/k20proj/data/adddosdn_cicflow_dataset.csv",
}
```

Dataset assumptions:

- There is a **label column** (case-insensitive) among:
  - `label`, `class`, `attack_type`, `attack type`
- All remaining **numeric columns** are treated as input features.
- Rows with `NaN`, `inf`, or `-inf` are dropped before training.

Labels are encoded using `LabelEncoder` and treated as multi-class targets.

---

## Methodology

### Global Configuration

Key configuration constants:

- `RANDOM_STATE = 42`
- `TEST_SIZE = 0.30` (stratified 70/30 train/test split)
- `MAX_SEQ_LEN = 128` (maximum sequence length for tokenization)
- `PCA_VARIANCE = 0.95` (retain 95% of variance, number of components is data-dependent)
- `N_FOLDS = 5` (for Stratified K-Fold CV on the hybrid models)
- `LR_MAX_ITER = 1000`
- Device auto-detection: GPU if available, else CPU.

### SLM Embedding Optimization

The notebook includes several optimizations for faster embedding extraction:

- **FP16 Precision**: Uses half precision (`USE_FP16 = True`) for faster inference on GPU with minimal accuracy loss.
- **Batch Processing**: Configurable batch size (`EMB_BATCH_SIZE = 128`) for better GPU utilization.
- **Model Compilation**: Uses `torch.compile()` (`ENABLE_COMPILE = True`) for PyTorch 2.0+ optimized inference.
- **Embedding Caching**: Caches embeddings to disk (`CACHE_EMBEDDINGS = True`) to avoid recomputation across runs.

### Data Loading and Preprocessing

1. **Loading** (`load_data`):
   - Reads each CSV into a DataFrame.
   - Strips column names.
   - Detects the label column.
   - Replaces `±inf` with `NaN` and drops `NaN` rows.
   - Selects numeric feature columns and encodes labels as integers.

2. **Train/Test Split**:
   - Uses `train_test_split` with stratification on labels (`test_size = 0.30`, `random_state = 42`).

3. **Tabular Preprocessing** (`preprocess_tabular`):
   - Standardizes features with `StandardScaler`.
   - Applies **SMOTE** on the training split only, with `k_neighbors` chosen adaptively:
     - `k_neighbors = min(5, min_class_count - 1)`, clipped to at least 1.
   - Applies **PCA** to the SMOTE-resampled training data, retaining 95% variance.
   - Transforms standardized test data with the same PCA model.

4. **Non-SMOTE PCA for Hybrid/SLM**:
   - A separate scaler and PCA are fit on **non-SMOTE** train data to obtain PCA features aligned 1:1 with the original rows, for use with semantic embeddings and the hybrid fusion.

### Semantic Embeddings (SLM)

- **Prompt construction** (`build_prompts`):
  - Each numeric feature vector is converted into a structured string resembling a Python dict:
    - Example: `{"' Feature1': 0.123, ' Feature2': -1.234, ...}`
  - This format matches the structure used to fine-tune DDoSBert.

- **Embedding extraction** (`extract_slm_embeddings`):
  - Uses `AutoTokenizer` and `AutoModel` from the HuggingFace model `Thi-Thu-Huong/DDoSBert`.
  - Tokenizes prompts with truncation/padding to `MAX_SEQ_LEN = 128`.
  - Computes a **mask-weighted mean pooling** over the last hidden state to obtain one embedding per sample.
  - Executes in batches on GPU if available.
  - Supports caching to disk to avoid recomputation.
  - Optimized with FP16 and model compilation for faster inference.

The resulting arrays `E_tr` and `E_te` are semantic embeddings of train and test samples.

---

## Model Variants

All classifiers use `LogisticRegression` with:

- `solver="lbfgs"`
- `multi_class="auto"`
- `C=1.0`
- `max_iter=LR_MAX_ITER`
- `n_jobs=-1`

Metrics are computed using **macro averaging**:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score

Inference time per sample (ms) is also reported.

### (A) Baseline: PCA + LR

- Input: SMOTE-resampled PCA features (`X_tr_pca`), original test PCA features (`X_te_pca`).
- Training: LR on `X_tr_pca` vs. resampled labels.
- Evaluation: Metrics on the original 30% test split.

### (B) SLM-only: Embeddings + LR (SMOTE)

- Input: Semantic embeddings `E_tr` (train) and `E_te` (test).
- SMOTE: Applied on `E_tr` using adaptive `k_neighbors` computed from raw training labels.
- Training: LR on SMOTE-resampled embeddings.
- Evaluation: Metrics on `E_te` vs. original test labels.

### (C) Hybrid: Attention Fusion → LR (SMOTE)

- Inputs:
  - Non-SMOTE PCA features (`X_tr_pca_ns`, `X_te_pca_ns`)
  - Semantic embeddings (`E_tr`, `E_te`)
- Fusion network (`AttentionFusionMLP`):
  - Concatenates PCA features `z` and embeddings `e`.
  - Applies a small MLP to produce a scalar gate `w ∈ (0,1)` per sample.
  - Outputs concatenated `[w * z, (1 - w) * e]`.
- Training:
  - The fusion module is trained with a reconstruction objective (matching concatenated `[z, e]`).
  - Training uses 5 epochs, Adam optimizer with learning rate `1e-3`, and batch size 512.
  - After training, fused features (`F_tr`, `F_te`) are computed.
  - SMOTE is applied to `F_tr` with adaptive `k_neighbors`.
  - LR is trained on the SMOTE-resampled fused features.
- Evaluation: Metrics on `F_te` vs. original test labels.

### (D) Hybrid: Concatenation → LR (SMOTE)

- Inputs:
  - Non-SMOTE PCA features (`X_tr_pca_ns`, `X_te_pca_ns`)
  - Semantic embeddings (`E_tr`, `E_te`)
- Fusion method (`fusion_concatenation`):
  - Simple concatenation: `[PCA features | SLM embeddings]`
- Training:
  - SMOTE is applied to concatenated features `F_tr` with adaptive `k_neighbors`.
  - LR is trained on the SMOTE-resampled concatenated features.
- Evaluation: Metrics on `F_te` vs. original test labels.

---

## Cross-Validation and Visualization

### 5-Fold Stratified Cross-Validation

Two cross-validation functions are provided:

1. **`run_hybrid_kfold`**: 5-fold CV for Attention Fusion variant
2. **`run_concat_kfold`**: 5-fold CV for Concatenation variant

Both functions:

- Use **StratifiedKFold** on the training portion.
- For each fold:
  - Trains a fusion module (Attention Fusion) or concatenates features (Concatenation).
  - Applies SMOTE inside the training fold with adaptive `k_neighbors`.
  - Trains Logistic Regression on the fused/concatenated features.
  - Evaluates on the validation fold.
- Reports:
  - Per-fold accuracy, precision, recall, F1-score (macro-averaged).
  - Mean ± standard deviation summary across the 5 folds.

### Confusion Matrices

Function: `plot_confusion_matrices(y_test, predictions, class_names, dataset_name)`

- Plots confusion matrices for each variant (A, B, C, D) on the test set.
- Uses `ConfusionMatrixDisplay` from scikit-learn.
- Displays all matrices side-by-side for easy comparison.

---

## Notebook Usage

1. **Open the notebook** `Final.ipynb` in Jupyter, VS Code, or Colab.
2. **Adjust dataset paths** in the `DATASETS` dictionary to point to your CSV files.
3. (Optional) Configure a HuggingFace token in your environment to avoid download throttling.
4. **Run all cells in order**:
   - The notebook will:
     - Install required libraries (if needed).
     - Detect CPU/GPU and enable optimizations (FP16, model compilation).
     - Load and preprocess datasets.
     - Extract semantic embeddings (with caching support).
     - Train and evaluate all four variants (A, B, C, D).
     - Run 5-fold CV for both Attention Fusion and Concatenation models.
     - Generate confusion matrices.
     - Print a consolidated summary table across datasets and variants.

Because SMOTE and embedding extraction can be computationally expensive on large datasets, expect non-trivial runtimes. The embedding caching feature helps reduce recomputation time on subsequent runs.

---

## Interpretation and Extensions

The notebook is structured to support the following analyses:

- **Ablation** between:
  - Purely statistical (PCA + LR),
  - Purely semantic (SLM-only + LR),
  - Hybrid fused representations (Attention Fusion and Concatenation).
- **Robustness** via 5-fold cross-validation on both hybrid variants.
- **Qualitative understanding** via confusion matrices.
- **Performance comparison** between Attention Fusion and simple Concatenation fusion strategies.

Potential extensions (not implemented here but supported by the structure):

- Alternative classifiers (e.g., tree-based, neural networks) on top of the same features.
- Different imbalance handling strategies (e.g., class weighting, other resamplers).
- Alternative fusion architectures (e.g., deeper MLPs, transformer-based fusion).
- Cross-dataset transfer (train on one dataset, test on another).
- Hyperparameter tuning for the fusion network and classifier.

---

## Reproducibility Notes

- A fixed random seed (`RANDOM_STATE = 42`) is used for:
  - Train/test splitting,
  - SMOTE resampling,
  - KFold splitting,
  - Random permutations in fusion training.
- PCA retains a fixed fraction of variance (95%), so the actual number of components is dataset-dependent but deterministic given the data and seed.
- DDoSBert parameters are loaded from HuggingFace and used as a **frozen encoder**; no fine-tuning is performed in this notebook.
- Embedding caching ensures that embeddings are consistent across runs when using the same data splits.
