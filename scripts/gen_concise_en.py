import pandas as pd
import tiktoken
from tqdm import tqdm
import logging
import argparse
import os
import sys
import time
import re
import yaml
from openai import OpenAI

# ─────── OpenRouter API Setup ───────
def init_client(api_key: str):
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

# ─────── Tokenizer Setup ───────
enc = tiktoken.encoding_for_model("gpt-4o-mini")

# ─────── Config Loader ───────
def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# ─────── Prompt Builder ───────
def build_concise_prompt(question, choices, original_explanation, target_words, prediction):
    return f"""
You are given a multiple-choice question, its options, and a previous explanation.

Your task:
- Keep the predicted answer fixed: {prediction}.
- Rewrite the explanation in a shorter, more concise version.
- Limit the rewritten explanation to about {target_words} words (±2 words).
- Do NOT mention option letters (A/B/C/D) in the explanation.
- Do NOT copy option texts or keywords directly.
- Focus only on the reasoning behind the correct choice.

Question:
{question}

Options:
A) {choices['A']}
B) {choices['B']}
C) {choices['C']}
D) {choices['D']}

Original Explanation:
{original_explanation}

Output Format:
<answer>{prediction}</answer>
<concise_explanation>
[Rewritten concise reasoning here — no option labels or copy-pasted keywords]
</concise_explanation>
""".strip()

# ─────── API Call ───────
def rewrite_explanation(client, model_name, question, choices, original_explanation, target_words, prediction):
    prompt = build_concise_prompt(question, choices, original_explanation, target_words, prediction)
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise scientific assistant. "
                        "Your job is to rewrite explanations into concise versions while "
                        "keeping them sufficient to justify the same predicted answer. "
                        "Never change the predicted answer. "
                        "Never add new facts. "
                        "Only shorten the explanation by removing redundancy."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        output = completion.choices[0].message.content.strip()

        ans_match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
        exp_match = re.search(r"<concise_explanation>(.*?)</concise_explanation>", output, re.DOTALL)

        ans = ans_match.group(1).strip() if ans_match else prediction
        exp = exp_match.group(1).strip() if exp_match else output

        return ans, exp, output
    except Exception as e:
        logging.error(f"Rewrite failed: {e}")
        return prediction, "", ""

# ─────── Helpers ───────
def count_words(text):
    return len(text.split())

# ─────── Main Evaluation Function ───────
def evaluate_explanation_conciseness(df, client, cfg):
    percentages = cfg.get("percentages", [90, 80, 70, 60, 50, 40, 30, 20, 10])

    for pct in percentages:
        output_rows = []
        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Length Constraint {100 - pct}%"):
            full_expl = row["explanation"]
            total_words = count_words(full_expl)

            question = row["question"]
            choices = {
                "A": row["choice_A"],
                "B": row["choice_B"],
                "C": row["choice_C"],
                "D": row["choice_D"]
            }
            prediction = row["prediction"]
            target_words = max(1, int(total_words * pct / 100))

            ans, concise_expl, raw_output = rewrite_explanation(
                client,
                cfg["model"],
                question,
                choices,
                full_expl,
                target_words,
                prediction
            )
                    # --- Logging outputs ---
            if i < 3:  # log first 3 rows fully
                logging.info(
                    f"[Row {i}] {100 - pct}% constraint → original {total_words} | "
                    f"target {target_words} words | Rewritten: {concise_expl[:60]}..."
                )
            elif (i + 1) % 100 == 0:  # log every 100th row briefly
                logging.info(
                    f"[Row {i}] {100 - pct}% constraint → original {total_words} | "
                    f"target {target_words} words | Rewritten: {concise_expl[:60]}..."
                )

            output_rows.append({
                "row_id": row["row_id"],
                "question_id": row["question_id"],
                "actual_answer": row["actual_answer"],
                "prediction": prediction,
                "is_correct": row["is_correct"],
                "split": row["split"],
                "question": question,
                "choice_A": row["choice_A"],
                "choice_B": row["choice_B"],
                "choice_C": row["choice_C"],
                "choice_D": row["choice_D"],
                "original_words": total_words,
                "length_constraint_percent": pct,
                "target_word_limit": target_words,
                "original_explanation": full_expl,
                "rewritten_answer": ans,
                "rewritten_explanation": concise_expl,
                "rewritten_word_count": count_words(concise_expl),
                "raw_output": raw_output,
            })

            time.sleep(cfg.get("sleep", 0.7))  # avoid hitting rate limit

        # Save results
        out_path = os.path.join(cfg["output_dir"], f"concise_rewrites_{cfg['model_name']}_{pct}.csv")
        os.makedirs(cfg["output_dir"], exist_ok=True)
        pd.DataFrame(output_rows).to_csv(out_path, index=False)
        logging.info(f"✅ Saved {100 - pct}% rewrites → {out_path}")

# ─────── Main ───────
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

    client = init_client(cfg.get("api_key", os.getenv("OPENROUTER_API_KEY")))
    evaluate_explanation_conciseness(df, client, cfg)

if __name__ == "__main__":
    main()
