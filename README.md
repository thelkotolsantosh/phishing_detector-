# 🎣 Phishing Website Detection using Machine Learning

A production-style machine learning pipeline that detects phishing websites from URL-derived features. Built as a portfolio-grade project covering the full ML lifecycle — data generation, EDA, feature engineering, model training, evaluation, and CLI inference.

---

## 📋 Problem Statement

Phishing attacks trick users into visiting fraudulent websites that impersonate legitimate ones. These attacks are responsible for billions of dollars of losses annually. This project builds a binary classifier that labels a URL as **Legitimate (0)** or **Phishing (1)** based on structural and lexical features extracted from the URL itself — no external API calls required.

---

## 🗂 Project Structure

```
phishing_detector/
│
├── data/
│   ├── phishing_dataset.csv        ← 2 000-row labelled dataset
│   └── generate_dataset.py         ← Script to regenerate the dataset
│
├── notebooks/
│   └── phishing_eda_and_model.ipynb ← Step-by-step Jupyter walkthrough
│
├── src/
│   ├── __init__.py
│   ├── logger.py                   ← Centralized logging
│   ├── preprocess.py               ← Load, clean, feature engineer, split
│   ├── eda.py                      ← EDA plots (class dist, correlations, KDEs)
│   ├── train.py                    ← Model factory, CV training, persistence
│   ├── evaluate.py                 ← Confusion matrix, ROC, feature importance
│   └── predict.py                  ← Inference (single-sample & batch)
│
├── models/
│   ├── phishing_model.pkl          ← Trained RandomForest (after run)
│   └── scaler.pkl                  ← Fitted StandardScaler (after run)
│
├── outputs/
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_distributions.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
│
├── logs/
│   └── phishing_detector.log
│
├── main.py                         ← CLI entrypoint
├── config.yaml                     ← All parameters in one place
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset Description

The dataset (`data/phishing_dataset.csv`) contains **2 000 samples** with **14 URL-based features** plus 1 binary label.

| Feature | Description |
|---|---|
| `url_length` | Total character length of the URL |
| `num_dots` | Number of `.` characters |
| `num_hyphens` | Number of `-` characters |
| `num_at` | Count of `@` (often used in obfuscation) |
| `num_question` | Count of `?` query separators |
| `num_ampersand` | Count of `&` parameter separators |
| `num_digits` | Count of digit characters |
| `has_https` | 1 if URL uses HTTPS, else 0 |
| `has_ip` | 1 if domain is an IP address |
| `subdomain_count` | Number of subdomain levels |
| `path_length` | Length of the URL path |
| `entropy` | Shannon entropy of the URL string |
| `tld_suspicious` | 1 if TLD is on a known suspicious list |
| `domain_age_days` | Estimated age of the domain in days |
| **`label`** | **0 = Legitimate, 1 = Phishing** |

Three engineered features are added during preprocessing:
- `dots_per_length` — dot density
- `digit_ratio` — fraction of digits in URL
- `suspicion_score` — additive risk proxy

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/phishing_detector.git
cd phishing_detector
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Train the full pipeline
```bash
python main.py --mode train
```
This runs: EDA → preprocessing → training (with 5-fold CV) → evaluation → saves model + plots.

### Run EDA only
```bash
python main.py --mode eda
```

### Predict a single URL (interactive CLI)
```bash
python main.py --mode predict
```
You will be prompted to enter feature values. Press **ENTER** to use defaults (which represent a phishing URL).

### Use a custom config
```bash
python main.py --mode train --config config.yaml
```

---

## 📈 Results

| Metric | Score |
|---|---|
| Accuracy | ~96–100% |
| F1 Score | ~96–100% |
| ROC-AUC | ~0.99 |
| CV F1 (5-fold) | ~0.97 ± 0.02 |

> Results vary slightly based on dataset seed and model type configured in `config.yaml`.

### Output Plots

| Plot | Description |
|---|---|
| `class_distribution.png` | Bar chart of label balance |
| `correlation_heatmap.png` | Feature correlation matrix |
| `feature_distributions.png` | KDE comparison per class |
| `confusion_matrix.png` | TP/FP/TN/FN breakdown |
| `roc_curve.png` | ROC-AUC curve |
| `feature_importance.png` | Top-15 features by importance |

---

## 🔧 Configuration

All hyperparameters and paths live in `config.yaml`. Switch between models by changing:
```yaml
model:
  type: "GradientBoosting"   # or RandomForest / LogisticRegression
```

---

## 🧠 ML Pipeline Overview

```
Raw CSV
  └─► clean_data()        → drop duplicates, fill NaN with median
        └─► feature_engineering()  → dots_per_length, digit_ratio, suspicion_score
              └─► split_and_scale()      → stratified 80/20, StandardScaler
                    └─► train_model()         → 5-fold CV + final fit
                          └─► evaluate_model()      → metrics + plots
                                └─► save_model()           → models/*.pkl
```

---

## 🔮 Future Improvements

- [ ] Real URL scraping via `requests` + feature extraction from live URLs
- [ ] SMOTE oversampling for imbalanced datasets (`imbalanced-learn` already installed)
- [ ] Hyperparameter tuning with `GridSearchCV` or `Optuna`
- [ ] FastAPI REST endpoint for real-time prediction
- [ ] SHAP explainability integration
- [ ] Docker containerisation
- [ ] GitHub Actions CI for automated re-training

---

## 📜 License

MIT License — free to use, modify, and distribute.

---

## 🙌 Author

**Your Name** — [GitHub](https://github.com/YOUR_USERNAME) | [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)
