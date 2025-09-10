import pandas as pd
import logging
from src.evaluation.explaination_evaluator import evaluate_models_with_masked_explanation
from src.utils.helpers import mask_explanation, mask_explanation_limit

# ───── Setup Logger ─────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("concise_eval")

# ───── Model to Evaluate ─────
model_names = [
    ("Qwen/Qwen3-1.7B", "qwen3-1.7b"),
]

# ───── Option Columns and Labels ─────
option_columns = ['choice_A', 'choice_B', 'choice_C', 'choice_D']
option_letters = ['A', 'B', 'C', 'D']

# ───── List of Trim Files and Their Percent Labels ─────
trim_versions = {
    0: "/home/a_zahedzadeh/self-explaination-thesis/results/models/prediction/gpt_4o_mini/en/pre_results_compelete_gpt_4o_mini_arc_challenge.csv",
    90: "/home/a_zahedzadeh/self-explaination-thesis/concise_rewrites_90.csv",
    80: "/home/a_zahedzadeh/self-explaination-thesis/concise_rewrites_80.csv",
    70: "/home/a_zahedzadeh/self-explaination-thesis/concise_rewrites_70.csv",
    60: "/home/a_zahedzadeh/self-explaination-thesis/concise_rewrites_60.csv",
    50: "/home/a_zahedzadeh/self-explaination-thesis/concise_rewrites_50.csv",
    40: "/home/a_zahedzadeh/self-explaination-thesis/concise_rewrites_40.csv",
    30: "/home/a_zahedzadeh/self-explaination-thesis/concise_rewrites_30.csv",
    20: "/home/a_zahedzadeh/self-explaination-thesis/concise_rewrites_20.csv",
    10: "/home/a_zahedzadeh/self-explaination-thesis/concise_rewrites_10.csv",

}

# ───── Run Evaluation Per Trim Level ─────
for pct, filepath in trim_versions.items():
    logger.info(f"\n===== Running evaluation for trim_percent = {pct}% =====")

    
    if pct == 0:
        df = pd.read_csv(filepath)
        #df = df.sample(n=100, random_state=42)
        df['masked_explanation'] = df.apply(mask_explanation, axis=1)
    else:
        df = pd.read_csv(filepath)
        df['masked_explanation'] = df.apply(mask_explanation_limit, axis=1)

    # ───── Apply masking before second evaluation ─────
    

    # Evaluate with masked explanation
    output_with_masked = evaluate_models_with_masked_explanation(
        model_names=model_names,
        df=df,
        option_columns=option_columns,
        option_letters=option_letters,
        logger=logger,
    )
    output_with_masked.to_csv(f"results_masked_{pct}.csv", index=False)
