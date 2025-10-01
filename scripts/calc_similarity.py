import os
import glob
import logging
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import yaml

# ---------------------------
# Logging setup
# ---------------------------
def setup_logging(level="INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # silence noisy libs
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    return logging.getLogger(__name__)

# ---------------------------
# Compute similarity (batched)
# ---------------------------
def compute_similarity(model, base_exps, df):
    base_texts = [base_exps.get(idx, "") for idx in df["index"]]
    var_texts = df["masked_explanation"].tolist()

    # batch encode base + variants in one go
    embeddings = model.encode(
        base_texts + var_texts,
        batch_size=32,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    n = len(df)
    base_embs = embeddings[:n]
    var_embs = embeddings[n:]

    # diagonal cos_sim
    sims = util.cos_sim(base_embs, var_embs).diagonal()
    return sims.tolist()

# ---------------------------
# Process a single folder (task)
# ---------------------------
def process_folder(folder, model, logger):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        logger.warning(f"No CSV files found in {folder}")
        return

    base_file = [f for f in files if f.endswith("_0.csv")]
    if not base_file:
        logger.error(f"No base file (_0.csv) found in {folder}")
        return
    base_file = base_file[0]

    base_df = pd.read_csv(base_file)
    base_exps = dict(zip(base_df["index"], base_df["masked_explanation"]))

    trimmed_files = [f for f in files if not f.endswith("_0.csv")]

    for f in tqdm(trimmed_files, desc=f"Processing {os.path.basename(folder)}", position=0, leave=True):
        df = pd.read_csv(f)
        df["similarity_with_base"] = compute_similarity(model, base_exps, df)
        df.to_csv(f, index=False)
        logger.info(f"✅ Updated {f}")

# ---------------------------
# Discover all task folders
# ---------------------------
def discover_tasks(root_dir):
    tasks = []
    for dirpath, _, filenames in os.walk(root_dir):
        if any(f.endswith("_0.csv") for f in filenames):
            tasks.append(dirpath)
    return tasks

# ---------------------------
# Main
# ---------------------------
def main(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    logger = setup_logging(cfg.get("logging", "INFO"))

    model_name = cfg["embedding_model"]
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    root_dir = cfg["root_dir"]
    tasks = discover_tasks(root_dir)
    logger.info(f"Discovered {len(tasks)} tasks under {root_dir}")

    for folder in tasks:
        logger.info(f"=== Processing task folder: {folder} ===")
        process_folder(folder, model, logger)

    logger.info("🎉 All tasks finished successfully.")

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute similarity of explanations with base version")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    main(args.config)
