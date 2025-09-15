import os
import re
import time
import logging
import argparse
import yaml
from typing import Tuple, List, Dict

import pandas as pd
import tiktoken
from tqdm import tqdm
from openai import OpenAI


# ─────── Helpers ───────
def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def count_words(text: str) -> int:
    return len(str(text).split())


def contains_persian(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", str(text)))


def extract_tags(xml_text: str, fallback_answer: str) -> Tuple[str, str]:
    xml_text = xml_text.strip()
    ans_match = re.search(r"<answer>(.*?)</answer>", xml_text, re.DOTALL | re.IGNORECASE)
    exp_match = re.search(
        r"<concise_explanation>(.*?)</concise_explanation>",
        xml_text,
        re.DOTALL | re.IGNORECASE,
    )

    ans = ans_match.group(1).strip() if ans_match else fallback_answer
    exp = exp_match.group(1).strip() if exp_match else xml_text
    return ans, exp


def build_concise_prompt(
    question: str,
    choices: Dict[str, str],
    original_explanation: str,
    target_words: int,
    prediction: str,
) -> str:
    return f"""
You are given a multiple-choice question, its options, and a previous explanation.

Your task:
- Keep the predicted answer fixed: {prediction}.
- Rewrite the explanation into a shorter, more concise version.
- The rewritten explanation must be written in **Persian (Farsi)**.
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

Output Format (return exactly these two tags):
<answer>{prediction}</answer>
<concise_explanation>
[Concise reasoning in Persian here — no option labels or copy-pasted keywords]
</concise_explanation>
""".strip()


def rewrite_explanation(
    client: OpenAI,
    question: str,
    choices: Dict[str, str],
    original_explanation: str,
    target_words: int,
    prediction: str,
    model_name: str,
    temperature: float,
    retry_if_not_persian: bool = True,
) -> Tuple[str, str, str]:
    base_messages = [
        {
            "role": "system",
            "content": (
                "You are a precise scientific assistant. "
                "Your job is to rewrite explanations into concise versions while keeping them sufficient to justify the same predicted answer. "
                "Never change the predicted answer. Never add new facts. "
                "Only shorten the explanation by removing redundancy. "
                "Important: the concise explanation must be written in Persian (Farsi). "
                "Return output ONLY using the two XML-like tags <answer> and <concise_explanation>."
            ),
        },
        {
            "role": "user",
            "content": build_concise_prompt(
                question, choices, original_explanation, target_words, prediction
            ),
        },
    ]

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=base_messages,
            temperature=temperature,
        )
        output = completion.choices[0].message.content.strip()
        ans, exp = extract_tags(output, fallback_answer=prediction)

        if retry_if_not_persian and not contains_persian(exp):
            stricter_messages = base_messages + [
                {
                    "role": "user",
                    "content": (
                        "Reminder: The concise explanation must be written in Persian (Farsi). "
                        "Return only the two tags exactly."
                    ),
                }
            ]
            completion2 = client.chat.completions.create(
                model=model_name,
                messages=stricter_messages,
                temperature=temperature,
            )
            output2 = completion2.choices[0].message.content.strip()
            ans2, exp2 = extract_tags(output2, fallback_answer=ans)
            if contains_persian(exp2):
                return ans2, exp2, output2

        return ans, exp, output

    except Exception as e:
        logging.error(f"Rewrite failed: {e}")
        return prediction, "", ""


def evaluate_explanation_conciseness(
    df: pd.DataFrame,
    client: OpenAI,
    cfg: dict,
) -> None:
    percentages = cfg.get("percentages", [90, 80, 70, 60, 50, 40, 30, 20, 10])
    model_name = cfg.get("model", "openai/gpt-4o-mini")
    temperature = cfg.get("temperature", 0.3)
    sleep_seconds = cfg.get("sleep_seconds", 1.0)

    save_dir = cfg.get("save_dir", ".")
    save_prefix = cfg.get("save_prefix", "concise_rewrites")
    os.makedirs(save_dir, exist_ok=True)

    for pct in percentages:
        output_rows = []
        desc = f"Length Constraint {100 - pct}% (keep ~{pct}%)"

        for i, row in tqdm(df.iterrows(), total=len(df), desc=desc):
            full_expl = str(row.get("explanation", ""))
            total_words = count_words(full_expl)

            question = str(row.get("question", ""))
            choices = {
                "A": str(row.get("choice_A", "")),
                "B": str(row.get("choice_B", "")),
                "C": str(row.get("choice_C", "")),
                "D": str(row.get("choice_D", "")),
            }
            prediction = str(row.get("prediction", ""))

            target_words = max(1, int(total_words * pct / 100))

            ans, concise_expl, raw_output = rewrite_explanation(
                client,
                question,
                choices,
                full_expl,
                target_words,
                prediction,
                model_name,
                temperature,
            )

            logging.info(
                f"[Row {i}] {100 - pct}% cut → original {total_words} | target {target_words} | rewritten: {concise_expl[:60]}..."
            )

            output_rows.append(
                {
                    "row_id": row.get("row_id", i),
                    "question_id": row.get("question_id", ""),
                    "actual_answer": row.get("actual_answer", ""),
                    "prediction": prediction,
                    "is_correct": row.get("is_correct", ""),
                    "split": row.get("split", ""),
                    "question": question,
                    "choice_A": choices["A"],
                    "choice_B": choices["B"],
                    "choice_C": choices["C"],
                    "choice_D": choices["D"],
                    "original_words": total_words,
                    "length_constraint_percent": pct,
                    "target_word_limit": target_words,
                    "original_explanation": full_expl,
                    "rewritten_answer": ans,
                    "rewritten_explanation": concise_expl,
                    "rewritten_word_count": count_words(concise_expl),
                    "raw_output": raw_output,
                }
            )

            time.sleep(sleep_seconds)

        save_path = os.path.join(save_dir, f"{save_prefix}_{pct}.csv")
        pd.DataFrame(output_rows).to_csv(save_path, index=False)
        logging.info(f"✅ Saved rewrites → {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    logging.basicConfig(level=getattr(logging, cfg.get("log_level", "INFO").upper()), format="%(asctime)s - %(message)s")

    # Setup client
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=cfg.get("api_key", os.getenv("OPENROUTER_API_KEY")),
    )

    df = pd.read_csv(cfg["input"])
    logging.info(f"Loaded dataset with {len(df)} rows from {cfg['input']}")

    evaluate_explanation_conciseness(df, client, cfg)
