import os
import re
import time
import logging
from typing import Tuple, List, Dict

import pandas as pd
import tiktoken
from tqdm import tqdm
from openai import OpenAI


# ─────── OpenRouter API Setup ───────
# Put your key in the environment: export OPENROUTER_API_KEY="sk-or-..."
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", "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"),
)

# ─────── Tokenizer Setup (optional; not strictly needed for word count) ───────
# Using word-based constraint for Persian; tokenizer left as optional utility.
try:
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
except Exception:
    enc = None


# ─────── Helpers ───────
def count_words(text: str) -> int:
    """Simple whitespace-based word count (sufficient for Persian)."""
    return len(str(text).split())


def contains_persian(text: str) -> bool:
    """Heuristic: checks if text has Persian/Arabic letters."""
    return bool(re.search(r"[\u0600-\u06FF]", str(text)))


def extract_tags(xml_text: str, fallback_answer: str) -> Tuple[str, str]:
    """
    Extracts <answer> and <concise_explanation> from model output.
    Falls back to provided answer or whole output if tags are missing.
    """
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


# ─────── Prompt Template (English instructions; Persian output) ───────
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


# ─────── API Call ───────
def rewrite_explanation(
    question: str,
    choices: Dict[str, str],
    original_explanation: str,
    target_words: int,
    prediction: str,
    model_name: str = "openai/gpt-4o-mini",
    temperature: float = 0.3,
    retry_if_not_persian: bool = True,
) -> Tuple[str, str, str]:
    """
    Calls the model to rewrite explanation.
    Returns: (answer, concise_explanation, raw_output)
    Retries once with a stricter reminder if the explanation is not Persian.
    """
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

        # If explanation isn't Persian, retry once with a stricter reminder
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


# ─────── Main Evaluation Function ───────
def evaluate_explanation_conciseness(
    df: pd.DataFrame,
    percentages: List[int] = None,
    model_name: str = "openai/gpt-4o-mini",
    sleep_seconds: float = 1.2,
) -> None:
    """
    For each percentage p in `percentages`, rewrites explanations to ~p% of original word count.
    Saves one CSV per percentage: concise_rewrites_{p}.csv
    """
    if percentages is None:
        percentages = [90, 80, 70, 60, 50, 40, 30, 20, 10]

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
            prediction = str(row.get("prediction", ""))  # keep A/B/C/D label

            # Compute target word count
            target_words = max(1, int(total_words * pct / 100))

            # Call model
            ans, concise_expl, raw_output = rewrite_explanation(
                question,
                choices,
                full_expl,
                target_words,
                prediction,
                model_name=model_name,
            )

            logging.info(
                f"[Row {i}] constraint {100 - pct}% → original {total_words} | target {target_words} | rewritten: {concise_expl[:60]}..."
            )

            output_rows.append(
                {
                    # Original metadata (use .get to avoid KeyError)
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

                    # Original explanation stats
                    "original_words": total_words,
                    "length_constraint_percent": pct,
                    "target_word_limit": target_words,
                    "original_explanation": full_expl,

                    # Rewritten explanation
                    "rewritten_answer": ans,
                    "rewritten_explanation": concise_expl,
                    "rewritten_word_count": count_words(concise_expl),
                    "raw_output": raw_output,
                }
            )

            # Be gentle with rate limits
            time.sleep(sleep_seconds)

        # Save per-constraint CSV
        output_df = pd.DataFrame(output_rows)
        output_df.to_csv(f"concise_rewrites_fa_{pct}.csv", index=False)
        logging.info(
            f"✅ Saved {100 - pct}% length-constrained rewrites → concise_rewrites_fa_{pct}.csv"
        )


# ─────── Example usage ───────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    # Set your CSV path here:
    # Example FA (Persian) dataset path; change as needed
    df_path = "/home/a_zahedzadeh/self-explaination-thesis/results/models/prediction/gpt_4o_mini/fa/pre_results_compelete_persian_gpt_4o_mini_arc_challenge.csv"

    df = pd.read_csv(df_path)

    # Choose model available on OpenRouter that supports Persian well
    # Examples (adjust to what you have access to):
    # model_name = "openai/gpt-4o-mini"
    # model_name = "qwen/qwen2.5-7b-instruct"
    # model_name = "google/gemma-3-4b-it"
    model_name = "openai/gpt-4o-mini"

    evaluate_explanation_conciseness(df, model_name=model_name)
