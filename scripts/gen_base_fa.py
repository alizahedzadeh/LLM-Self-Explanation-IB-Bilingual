import pandas as pd
import logging
import argparse
import os
import sys
import time
import yaml  # pip install pyyaml
from tqdm import tqdm  # pip install tqdm
from typing import Tuple
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.openrouter import OpenRouterModel
from src.utils.prompts import PREDICTION_BASE_PROMPT_TEMPLATE_PERSIAN


def load_config(config_path: str):
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_prompt(row, template):
    return template.format(
        question=row["question"],
        option_A=row["choice_A"],
        option_B=row["choice_B"],
        option_C=row["choice_C"],
        option_D=row["choice_D"],
    )


def parse_output(raw_output: str) -> Tuple[str, str]:
    """
    Extract prediction and explanation from raw model output using regex.
    Returns (prediction, explanation) as strings (empty string if missing).
    """
    try:
        pred_match = re.search(r"<prediction>\s*([A-D])\s*</prediction>", raw_output, re.IGNORECASE)
        prediction = pred_match.group(1).upper() if pred_match else ""

        expl_match = re.search(r"<explanation>\s*(.*?)\s*</explanation>", raw_output, re.DOTALL | re.IGNORECASE)
        explanation = expl_match.group(1).strip() if expl_match else ""

        return prediction, explanation

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error extracting prediction/explanation: {e}")
        return "", ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    logging.basicConfig(
        level=getattr(logging, cfg.get("log_level", "INFO").upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    df = pd.read_csv(cfg["input"])
    logging.info(f"Loaded dataset with {len(df)} rows from {cfg['input']}")

    model = OpenRouterModel(
        api_key=cfg.get("api_key", os.getenv("OPENROUTER_API_KEY")),
        model_name=cfg["model"],
        temperature=cfg.get("temperature", 0.3),
        max_tokens=cfg.get("max_tokens", 512),
    )

    results = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing", unit="row"):
        start_time = time.time()
        prompt = build_prompt(row, PREDICTION_BASE_PROMPT_TEMPLATE_PERSIAN)

        raw_reply, prediction, explanation = "", "", ""
        try:
            raw_reply = model.chat([
                {"role": "system", "content": cfg.get("system_prompt", "")},
                {"role": "user", "content": prompt},
            ])
            prediction, explanation = parse_output(raw_reply)
        except Exception as e:
            logging.error(f"Failed on row {i}: {e}")

        processing_time = round(time.time() - start_time, 2)
        is_correct = 1 if prediction == row["actual_answer"] else 0

        results.append({
            "row_id": i,
            "question_id": row.get("question_id", i),
            "prediction": prediction,
            "explanation": explanation,
            "raw_output": raw_reply,
            "actual_answer": row["actual_answer"],
            "is_correct": is_correct,
            "split": row.get("split", "test"),
            "processing_time": processing_time,
            "question": row["question"],
            "choice_A": row["choice_A"],
            "choice_B": row["choice_B"],
            "choice_C": row["choice_C"],
            "choice_D": row["choice_D"],
            "masked_explanation": row.get("masked_explanation", ""),
        })

        if i < 3:
            logging.info(
                f"Row {i} | Gold={row['actual_answer']} | Pred={prediction} "
                f"| Expl={explanation} | Raw={raw_reply}"
            )
        elif (i + 1) % 100 == 0:
            logging.info(
                f"Row {i+1}/{len(df)} | Gold={row['actual_answer']} | Pred={prediction} "
                f"| Expl snippet: {explanation[:80]}..."
            )
        time.sleep(cfg.get("sleep", 0.7))  # avoid hitting rate limit

    out_df = pd.DataFrame(results)
    cols = [
        "row_id", "question_id", "prediction", "explanation", "raw_output",
        "actual_answer", "is_correct", "split", "processing_time",
        "question", "choice_A", "choice_B", "choice_C", "choice_D",
        "masked_explanation"
    ]
    out_df = out_df[cols]

    # --- Save results ---
    save_dir = cfg.get("save_dir", ".")
    save_name = cfg.get("save_name", "results.csv")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, save_name)
    out_df.to_csv(save_path, index=True)
    logging.info(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()
