"""
main.py
───────
CLI entrypoint for the Phishing Website Detection pipeline.

Modes
─────
  python main.py --mode train      → full train + evaluate pipeline
  python main.py --mode predict    → single-sample interactive prediction
  python main.py --mode eda        → run EDA plots only

Usage examples
──────────────
  python main.py --mode train
  python main.py --mode predict
  python main.py --mode eda
  python main.py --mode train --config config.yaml
"""

import argparse
import yaml
import os
import sys

from src.logger     import get_logger
from src.preprocess import load_data, clean_data, feature_engineering, split_and_scale
from src.train      import get_model, train_model, save_model
from src.evaluate   import evaluate_model, plot_feature_importance
from src.eda        import class_distribution, correlation_heatmap, feature_distributions
from src.predict    import load_artifacts, predict_single

log = get_logger("main")

BANNER = r"""
╔══════════════════════════════════════════════════════╗
║   🎣  Phishing Website Detector — ML Pipeline       ║
║       Built with scikit-learn | Portfolio Project   ║
╚══════════════════════════════════════════════════════╝
"""


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    log.info(f"Config loaded from: {path}")
    return cfg


# ── Pipeline modes ──────────────────────────────────────────────────────────────

def run_eda(cfg: dict):
    """Generate all EDA plots from the raw dataset."""
    df = load_data(cfg["data"]["raw_path"])
    df = clean_data(df)
    df = feature_engineering(df)

    out = cfg["output"]["reports_dir"]
    class_distribution(df, target=cfg["features"]["target"], output_dir=out)
    correlation_heatmap(df, output_dir=out)
    feature_distributions(df, target=cfg["features"]["target"], output_dir=out)
    log.info("EDA complete — check outputs/ folder.")


def run_train(cfg: dict):
    """End-to-end training pipeline: load → clean → EDA → train → evaluate."""
    print(BANNER)

    # 1. Data
    df = load_data(cfg["data"]["raw_path"])
    df = clean_data(df)
    df = feature_engineering(df)

    # 2. EDA
    run_eda(cfg)

    # 3. Feature list (base + engineered)
    base_feats = cfg["features"]["numeric"] + cfg["features"]["binary"]
    extra_feats = ["dots_per_length", "digit_ratio", "suspicion_score"]
    all_features = base_feats + extra_feats
    target = cfg["features"]["target"]

    # 4. Split & scale
    X_train, X_test, y_train, y_test, _ = split_and_scale(
        df,
        feature_cols=all_features,
        target_col=target,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        scaler_path=cfg["model"]["scaler_path"],
    )

    # 5. Build & train model
    model_type = cfg["model"]["type"]
    params     = cfg["model"]["params"][model_type]
    model      = get_model(model_type, params)
    model      = train_model(model, X_train, y_train)

    # 6. Save model
    save_model(model, cfg["model"]["save_path"])

    # 7. Evaluate
    metrics = evaluate_model(model, X_test, y_test, cfg["output"]["reports_dir"])
    plot_feature_importance(model, all_features, cfg["output"]["reports_dir"])

    log.info("=" * 50)
    log.info(f"FINAL  Accuracy={metrics['accuracy']:.4f}  "
             f"F1={metrics['f1']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}")
    log.info("=" * 50)
    log.info("Training pipeline complete ✓")


def run_predict(cfg: dict):
    """Interactive single-sample prediction via CLI prompts."""
    model, scaler = load_artifacts(
        cfg["model"]["save_path"],
        cfg["model"]["scaler_path"],
    )

    base_feats  = cfg["features"]["numeric"] + cfg["features"]["binary"]
    extra_feats = ["dots_per_length", "digit_ratio", "suspicion_score"]
    all_features = base_feats + extra_feats

    print("\n── Enter URL features for prediction (press ENTER for defaults) ──\n")

    defaults = {
        "url_length": 120, "num_dots": 5, "num_hyphens": 4, "num_at": 1,
        "num_question": 2, "num_ampersand": 3, "num_digits": 8, "has_https": 0,
        "has_ip": 1, "subdomain_count": 3, "path_length": 60, "entropy": 4.2,
        "tld_suspicious": 1, "domain_age_days": 12,
        "dots_per_length": 5/121, "digit_ratio": 8/121, "suspicion_score": 4,
    }

    sample = {}
    for feat in all_features:
        raw = input(f"  {feat} [{defaults.get(feat, 0)}]: ").strip()
        sample[feat] = float(raw) if raw else float(defaults.get(feat, 0))

    result = predict_single(sample, model, scaler, all_features,
                            threshold=cfg["threshold"])
    print(f"\n  Result → {result['label']}")
    print(f"  Phishing probability : {result['probability']:.2%}\n")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phishing Website Detector — ML Pipeline"
    )
    parser.add_argument(
        "--mode", choices=["train", "predict", "eda"],
        default="train", help="Pipeline mode to execute (default: train)"
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: config.yaml)"
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = load_config(args.config)

    os.makedirs(cfg["output"]["reports_dir"], exist_ok=True)
    os.makedirs(cfg["output"]["logs_dir"],    exist_ok=True)

    mode_map = {
        "train":   run_train,
        "predict": run_predict,
        "eda":     run_eda,
    }
    mode_map[args.mode](cfg)


if __name__ == "__main__":
    main()
