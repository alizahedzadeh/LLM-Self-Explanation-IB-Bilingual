import pandas as pd
import logging
import argparse
import yaml
import os,sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.evaluation.explaination_evaluator import evaluate_models_with_masked_explanation
from src.utils.helpers import mask_explanation, mask_explanation_limit


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to eval config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger("concise_eval")

    model_names = [(m["name"], m["short"]) for m in cfg["models"]]
    option_columns = cfg["option_columns"]
    option_letters = cfg["option_letters"]
    trim_versions = cfg["trim_versions"]

    results_dir = cfg.get("results_dir", "results")
    filename_pattern = cfg.get("filename_pattern", "{model}_masked_{pct}.csv")

    os.makedirs(results_dir, exist_ok=True)

    for model_name, short_name in model_names:
        logger.info(f"\n===== Evaluating model {model_name} ({short_name}) =====")

        for pct, filepath in trim_versions.items():
            logger.info(f"\n--- Trim level = {pct}% ---")

            df = pd.read_csv(filepath)
            if pct == 0:
                df["masked_explanation"] = df.apply(mask_explanation, axis=1)
            else:
                df["masked_explanation"] = df.apply(mask_explanation_limit, axis=1)

            output_with_masked = evaluate_models_with_masked_explanation(
                model_names=[(model_name, short_name)],
                df=df,
                option_columns=option_columns,
                option_letters=option_letters,
                logger=logger,
            )

            filename = filename_pattern.format(model=short_name, pct=pct)
            out_file = os.path.join(results_dir, filename)
            output_with_masked.to_csv(out_file, index=False)
            logger.info(f"Saved results to {out_file}")


if __name__ == "__main__":
    main()
