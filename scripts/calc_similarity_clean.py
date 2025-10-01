import os
import glob
import logging
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import yaml

def setup_logging(level="INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    return logging.getLogger(__name__)

def compute_similarity(model, base_exps, df):
    base_texts = [base_exps.get(idx, "").replace("[MASK]", "") for idx in df["index"]]
    var_texts = [str(x).replace("[MASK]", "") for x in df["masked_explanation"].tolist()]

    embeddings = model.encode(
        base_texts + var_texts,
        batch_size=32,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    n = len(df)
    sims = util.cos_sim(embeddings[:n], embeddings[n:]).diagonal()
    return sims.tolist()

def process_folder(folder, model, logger):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    base_file = [f for f in files if f.endswith("_0.csv")][0]
    base_df = pd.read_csv(base_file)
    base_exps = dict(zip(base_df["index"], base_df["masked_explanation"]))
    trimmed_files = [f for f in files if not f.endswith("_0.csv")]

    for f in tqdm(trimmed_files, desc=f"Processing {os.path.basename(folder)}"):
        df = pd.read_csv(f)
        df["similarity_with_base_clean"] = compute_similarity(model, base_exps, df)
        df.to_csv(f, index=False)
        logger.info(f"✅ Updated {f}")

def discover_tasks(root_dir):
    tasks = []
    for dirpath, _, filenames in os.walk(root_dir):
        if any(f.endswith("_0.csv") for f in filenames):
            tasks.append(dirpath)
    return tasks

def main(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    logger = setup_logging(cfg.get("logging", "INFO"))
    model = SentenceTransformer(cfg["embedding_model"])
    tasks = discover_tasks(cfg["root_dir"])

    for folder in tasks:
        logger.info(f"=== Processing task: {folder} ===")
        process_folder(folder, model, logger)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
