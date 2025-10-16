# ================================================
# BERTScore Batch Pipeline (English / Persian)
# Stores all metrics: Precision / Recall / F1
# For both raw and [MASK]-cleaned text
# ================================================
# Usage:
#   python bertscore_pipeline.py --config config.yaml
# -----------------------------------------------
# Config example:
# root_dir: "./outputs/"
# language: "fa"    # or "en"
# logging: "INFO"
# ================================================

import os
import glob
import yaml
import logging
import pandas as pd
from tqdm import tqdm
from bert_score import score
from transformers import AutoConfig

# ---------------------------
# Logging setup
# ---------------------------
def setup_logging(level="INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    return logging.getLogger(__name__)


# ---------------------------
# BERTScore helper
# ---------------------------
def _infer_num_layers(model_name: str, fallback: int = 12) -> int:
    try:
        cfg = AutoConfig.from_pretrained(model_name)
        return getattr(cfg, "num_hidden_layers", fallback)
    except Exception:
        return fallback


def compute_bertscore_single(candidates, references, lang="en", verbose=False):
    """
    Compute BERTScore for aligned lists of candidates and references.
    Returns (Precision, Recall, F1) as lists.
    """
    if lang == "fa":
        model_type = "HooshvareLab/bert-base-parsbert-uncased"
        num_layers = _infer_num_layers(model_type, 12)
        kwargs = dict(
            model_type=model_type,
            num_layers=num_layers,
            lang=None,
            rescale_with_baseline=False,
            verbose=verbose
        )
    elif lang == "en":
        model_type = "roberta-large"
        num_layers = _infer_num_layers(model_type, 24)
        kwargs = dict(
            model_type=model_type,
            num_layers=num_layers,
            lang="en",
            rescale_with_baseline=False,
            verbose=verbose
        )
    else:
        raise ValueError("lang must be 'en' or 'fa'")

    P, R, F1 = score(candidates, references, **kwargs)
    return P.tolist(), R.tolist(), F1.tolist()


# ---------------------------
# Folder processing logic
# ---------------------------
def compute_bertscore_vs_base(base_exps, df, lang="en", logger=None):
    """
    Compare each explanation with its base version using BERTScore.
    Returns dictionaries for raw and clean text versions.
    """
    base_raw, var_raw = [], []
    base_clean, var_clean = [], []

    for idx in df["index"]:
        val = base_exps.get(idx, "")
        s = str(val) if isinstance(val, str) else ""
        base_raw.append(s)
        base_clean.append(s.replace("[MASK]", ""))

    for val in df["masked_explanation"]:
        s = str(val) if isinstance(val, str) else ""
        var_raw.append(s)
        var_clean.append(s.replace("[MASK]", ""))

    try:
        P_raw, R_raw, F1_raw = compute_bertscore_single(var_raw, base_raw, lang=lang)
        P_clean, R_clean, F1_clean = compute_bertscore_single(var_clean, base_clean, lang=lang)
    except Exception as e:
        if logger:
            logger.error(f"BERTScore computation failed: {e}")
        P_raw = R_raw = F1_raw = P_clean = R_clean = F1_clean = [0.0] * len(var_raw)

    return {
        "raw": {"P": P_raw, "R": R_raw, "F1": F1_raw},
        "clean": {"P": P_clean, "R": R_clean, "F1": F1_clean},
    }


def process_folder(folder, lang, logger):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    base_file = [f for f in files if f.endswith("_0_with_goldprob.csv")]
    if not base_file:
        logger.warning(f"No baseline (_0.csv) found in {folder}")
        return
    base_file = base_file[0]

    base_df = pd.read_csv(base_file)
    base_exps = dict(zip(base_df["index"], base_df["masked_explanation"]))

    trimmed_files = [f for f in files if not f.endswith("_0.csv")]

    for f in tqdm(trimmed_files, desc=f"Processing {os.path.basename(folder)}"):
        df = pd.read_csv(f)
        res = compute_bertscore_vs_base(base_exps, df, lang=lang, logger=logger)

        df["bertscore_P_raw"] = res["raw"]["P"]
        df["bertscore_R_raw"] = res["raw"]["R"]
        df["bertscore_F1_raw"] = res["raw"]["F1"]

        df["bertscore_P_clean"] = res["clean"]["P"]
        df["bertscore_R_clean"] = res["clean"]["R"]
        df["bertscore_F1_clean"] = res["clean"]["F1"]

        # میانگین برای لاگ
        mean_raw = sum(res["raw"]["F1"]) / len(res["raw"]["F1"]) if res["raw"]["F1"] else 0
        mean_clean = sum(res["clean"]["F1"]) / len(res["clean"]["F1"]) if res["clean"]["F1"] else 0
        logger.info(f"✅ {os.path.basename(f)} | mean F1 raw={mean_raw:.4f}, clean={mean_clean:.4f}")

        df.to_csv(f, index=False)


def discover_tasks(root_dir):
    tasks = []
    for dirpath, _, filenames in os.walk(root_dir):
        if any(f.endswith("_0_with_goldprob.csv") for f in filenames):
            tasks.append(dirpath)
    return tasks


# ---------------------------
# Main entry
# ---------------------------
def main(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    root_dir = cfg["root_dir"]
    lang = cfg.get("language", "en")
    logger = setup_logging(cfg.get("logging", "INFO"))

    logger.info(f"Starting BERTScore pipeline | Language={lang.upper()} | Root={root_dir}")

    tasks = discover_tasks(root_dir)
    if not tasks:
        logger.warning(f"No valid tasks found under {root_dir}")
        return

    for folder in tasks:
        logger.info(f"=== Processing task: {folder} ===")
        process_folder(folder, lang, logger)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
