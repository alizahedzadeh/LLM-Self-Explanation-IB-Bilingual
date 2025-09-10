import pandas as pd
import tiktoken
from tqdm import tqdm
import logging
import time
import re
from openai import OpenAI

# ─────── OpenRouter API Setup ───────
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",  # replace with your OpenRouter key
)
#os.environ["OPENROUTER_API_KEY"] = "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"
# ─────── Tokenizer Setup ───────
enc = tiktoken.encoding_for_model("gpt-4o-mini")

# ─────── Prompt Template ───────
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
def rewrite_explanation(question, choices, original_explanation, target_words, prediction):
    prompt = build_concise_prompt(question, choices, original_explanation, target_words, prediction)
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-4o-mini",
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

        # Extract <answer> and <concise_explanation>
        ans_match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL)
        exp_match = re.search(r"<concise_explanation>(.*?)</concise_explanation>", output, re.DOTALL)

        ans = ans_match.group(1).strip() if ans_match else prediction
        exp = exp_match.group(1).strip() if exp_match else output

        return ans, exp, output

    except Exception as e:
        logging.error(f"Rewrite failed: {e}")
        return prediction, "", ""

# ─────── Word Count Helper ───────
def count_words(text):
    return len(text.split())

# ─────── Main Evaluation Function ───────
def evaluate_explanation_conciseness(df, percentages=[90, 80, 70, 60, 50, 40, 30, 20, 10]):
    for pct in percentages:
        output_rows = []

        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Length Constraint {100 - pct}%"):
            full_expl = row['explanation']
            total_words = count_words(full_expl)

            question = row['question']
            choices = {
                "A": row['choice_A'],
                "B": row['choice_B'],
                "C": row['choice_C'],
                "D": row['choice_D']
            }
            prediction = row['prediction']

            # target word count
            target_words = max(1, int(total_words * pct / 100))

            # rewrite
            ans, concise_expl, raw_output = rewrite_explanation(
                question, choices, full_expl, target_words, prediction
            )

            logging.info(f"[Row {i}] {100 - pct}% constraint → orginal {total_words} | target {target_words} words | Rewritten: {concise_expl[:60]}...")

            output_rows.append({
                # Original metadata
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

                # Original explanation
                "original_words": total_words,
                "length_constraint_percent": pct,
                "target_word_limit": target_words,
                "original_explanation": full_expl,

                # Rewritten explanation
                "rewritten_answer": ans,
                "rewritten_explanation": concise_expl,
                "rewritten_word_count": count_words(concise_expl),
                "raw_output": raw_output
            })

            time.sleep(1.2)  # prevent API rate limit

        # Save per-constraint result
        output_df = pd.DataFrame(output_rows)
        output_df.to_csv(f"concise_rewrites_{pct}.csv", index=False)
        logging.info(f"✅ Saved {100 - pct}% length-constrained rewrites → concise_rewrites_{pct}.csv")

# ─────── Example usage ───────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    df = pd.read_csv("/home/a_zahedzadeh/self-explaination-thesis/results/models/prediction/gpt_4o_mini/en/pre_results_compelete_gpt_4o_mini_arc_challenge.csv")
    #sample_df = df.sample(n=100, random_state=42)
    evaluate_explanation_conciseness(df)
