import gc, torch, torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import os, re, time, logging

def evaluate_models_by_expl_types(model_names,
                                  df: pd.DataFrame,
                                  option_columns,
                                  option_letters,
                                  expl_variants,
                                  logger):
    """
    Compare multiple explanation types for each model.

    Args:
        model_names: list of (model_name, short_name)
        df: DataFrame with columns: question, choice_A..D, answerKey, and the explanation columns in `expl_variants`
        option_columns: e.g., ["choice_A", "choice_B", "choice_C", "choice_D"]
        option_letters: e.g., ["A", "B", "C", "D"]
        expl_variants: dict like {
            "noexp": None,                                # no explanation
            "masked_expl": "explanation_masked",          # masked original explanation
            "k1": "clauses_from_expl_k1_masked",          # masked K=1 block
            "k2": "clauses_from_expl_k2_masked",
            "k3": "clauses_from_expl_k3_masked",
            ...
        }
        logger: logging.Logger
    Returns:
        per_model_summaries: dict short_name -> summary DataFrame
    """

    def score_options(tokenizer, model, prompt):
        logprobs = []
        for l in option_letters:
            target = " " + l
            prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()[0]
            full_ids = tokenizer(prompt + target, return_tensors="pt").input_ids.cuda()[0]
            target_ids = full_ids[len(prompt_ids):]
            with torch.no_grad():
                outputs = model(full_ids.unsqueeze(0))
                logits = outputs.logits
            option_logits = logits[0, len(prompt_ids)-1:-1, :]
            log_probs = F.log_softmax(option_logits, dim=-1)
            total_log_prob = 0.0
            for j, token_id in enumerate(target_ids):
                total_log_prob += float(log_probs[j, token_id].item())
            logprobs.append(total_log_prob)
        probs = torch.softmax(torch.tensor(logprobs, dtype=torch.float32), dim=0)
        pred_idx = int(probs.argmax().item())
        return option_letters[pred_idx], float(probs[pred_idx].item()), probs

    per_model_summaries = {}

    for model_name, short_name in model_names:
        logger.info(f"\n=== Running model: {model_name} ({short_name}) ===")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        ).eval()

        # First compute the NO-EXPLANATION baseline once (used for sufficiency/usefulness deltas)
        baseline_rows = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"[{short_name}] Baseline (noexp)"):
            question = str(row["question"])
            options = [str(row[col]) for col in option_columns]
            gold = str(row["actual_answer"]).strip()

            prompt_noexp = (
                question + "\nOptions:\n" +
                "\n".join([f"{l}) {t}" for l, t in zip(option_letters, options)]) +
                "\nThe answer is "
            )
            if not prompt_noexp.endswith(" "): prompt_noexp += " "
            pred_letter, pred_conf, probs = score_options(tokenizer, model, prompt_noexp)

            baseline_rows.append({
                "index": row.get("row_id", idx),
                "pred_noexp": pred_letter,
                "prob_noexp": pred_conf,
                "is_correct_noexp": (pred_letter == gold),
                "probs_noexp": probs.cpu().numpy(),
            })
        baseline_df = pd.DataFrame(baseline_rows).set_index("index")

        # Evaluate each explanation type
        type_summaries = []
        for type_label, colname in expl_variants.items():
            logger.info(f"[{short_name}] Evaluating type: {type_label} (column={colname})")
            results = []

            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"[{short_name}] {type_label}"):
                qid = row.get("row_id", idx)
                question = str(row["question"])
                options = [str(row[col]) for col in option_columns]
                gold = str(row["actual_answer"]).strip()
                
                if colname is None:
                    # no explanation
                    explanation_text = ""
                else:
                    explanation_text = "" if pd.isna(row[colname]) else str(row[colname])

                if explanation_text.strip():
                    prompt = (
                        question + "\nExplanation: " + explanation_text.strip() +
                        "\nOptions:\n" +
                        "\n".join([f"{l}) {t}" for l, t in zip(option_letters, options)]) +
                        "\nThe answer is "
                    )
                else:
                    prompt = (
                        question + "\nOptions:\n" +
                        "\n".join([f"{l}) {t}" for l, t in zip(option_letters, options)]) +
                        "\nThe answer is "
                    )
                if not prompt.endswith(" "): prompt += " "

                pred_letter, pred_conf, probs = score_options(tokenizer, model, prompt)

                # fetch baseline
                base = baseline_df.loc[qid]
                pred_noexp = base["pred_noexp"]
                prob_noexp = float(base["prob_noexp"])
                is_correct_noexp = bool(base["is_correct_noexp"])

                is_correct_exp = (pred_letter == gold)

                # Sufficiency: explanation type leads to correct answer (common definition)
                sufficiency = is_correct_exp

                # Usefulness: either fixes a wrong baseline OR improves confidence while staying correct
                usefulness = ((not is_correct_noexp and is_correct_exp) or
                              (is_correct_noexp and is_correct_exp and (pred_conf > prob_noexp)))

                results.append({
                    "index": qid,
                    "type": type_label,
                    "question": question,
                    "gold": gold,
                    "pred_noexp": pred_noexp,
                    "prob_noexp": prob_noexp,
                    "pred_with_type": pred_letter,
                    "prob_with_type": pred_conf,
                    "is_correct_noexp": is_correct_noexp,
                    "is_correct_with_type": is_correct_exp,
                    "sufficiency": sufficiency,
                    "usefulness": usefulness,
                    "conf_delta": float(pred_conf - prob_noexp),
                })
                logger.info(
                        f"[{short_name}] Q{qid} | Type={type_label} | Gold={gold} "
                        f"| NoExp: {pred_noexp} ({prob_noexp:.3f}) "
                        f"| WithType: {pred_letter} ({pred_conf:.3f}) "
                        f"| CorrNoExp={is_correct_noexp} | CorrWithType={is_correct_exp} "
                        f"| Suff={int(sufficiency)} | Useful={int(usefulness)} "
                        f"| ΔConf={pred_conf - prob_noexp:+.3f}"
                    )

            results_df = pd.DataFrame(results)
            results_df.to_csv(f"results_{short_name}_{type_label}.csv", index=False)

            # Aggregate metrics for this type
            acc_noexp = results_df["is_correct_noexp"].mean()
            acc_with  = results_df["is_correct_with_type"].mean()
            suff_rate = results_df["sufficiency"].mean()
            useful    = results_df["usefulness"].mean()
            conf_gain = results_df["conf_delta"].mean()
            mean_suff = results_df["prob_with_type"].mean()


            logger.info(
                f"[{short_name}] Type={type_label} | "
                f"Acc(noexp)={acc_noexp:.3f} | Acc(type)={acc_with:.3f} | "
                f"Suff={suff_rate:.3f} | Useful={useful:.3f} | ΔConf={conf_gain:.3f}"
            )

            type_summaries.append({
                "type": type_label,
                "acc_noexp": acc_noexp,
                "acc_type": acc_with,
                "sufficiency_rate": suff_rate,
                "usefulness_rate": useful,
                "mean_conf_delta": conf_gain,
                "mean_sufficiency": mean_suff,
            })

        summary_df = pd.DataFrame(type_summaries).sort_values("type")
        summary_df.to_csv(f"summary_{short_name}.csv", index=False)
        per_model_summaries[short_name] = summary_df

        # Cleanup per model
        del model, tokenizer, baseline_df
        torch.cuda.empty_cache()
        gc.collect()

    return per_model_summaries


expl_variants = {
    "noexp": None,
    "masked_expl": "explanation_masked",
    "k2": "clauses_from_expl_k2_masked",
    "k3": "clauses_from_expl_k3_masked",
    "k4": "clauses_from_expl_k4_masked",
    "k5": "clauses_from_expl_k5_masked",
    "k6": "clauses_from_expl_k6_masked",
}
df_out = pd.read_csv("clauses_from_explanation.csv")
df = df_out.copy()

import re

def extract_clauses(text: str):
    """Extract plain clauses from a <reasoning> block."""
    return [c.strip() for c in re.findall(r"<r>(.*?)</r>", str(text), flags=re.DOTALL|re.IGNORECASE)]

# Apply to all clause columns
def expand_clause_columns(df, k_list=(2,3,4,5,6)):
    for K in k_list:
        col = f"clauses_from_expl_k{K}"
        if col in df.columns:
            new_col = f"clauses_list_k{K}"
            df[new_col] = df[col].apply(extract_clauses)
    return df

# Example usage
# df = pd.read_csv("your_results.csv")
df = expand_clause_columns(df, k_list=(2,3,4,5,6))

# Now you can see plain clauses as Python lists
print(df[["clauses_from_expl_k3", "clauses_list_k3"]].head())

# If you want each clause as its own column (clause_1, clause_2, …):
def expand_to_wide(df, k):
    col = f"clauses_list_k{k}"
    max_len = df[col].map(len).max()
    for i in range(max_len):
        df[f"clause_{i+1}_k{k}"] = df[col].map(lambda lst: lst[i] if i < len(lst) else "")
    return df

# Example: expand K=3
df = expand_to_wide(df, k=3)

import re
import pandas as pd

def mask_text(text: str, row) -> str:
    """Mask correct option letter and option texts inside a string."""
    if pd.isna(text):
        return ""
    out = str(text)

    # Mask correct option letter
    correct_letter = str(row["actual_answer"]).strip()
    if correct_letter:
        out = re.sub(rf"\b{correct_letter}\b", "[MASK]", out, flags=re.IGNORECASE)

    # Mask all option texts
    for col in ["choice_A", "choice_B", "choice_C", "choice_D"]:
        option_text = str(row[col])
        if option_text and option_text.lower() != "nan":
            out = re.sub(re.escape(option_text), "[MASK]", out, flags=re.IGNORECASE)

    return out

def apply_masking(df, k_list=(1,2,3,4,5)):
    # Mask the original explanation
    df["explanation_masked"] = df.apply(lambda r: mask_text(r["explanation"], r), axis=1)

    # Mask each clause column
    for K in k_list:
        col = f"clauses_from_expl_k{K}"
        if col in df.columns:
            new_col = f"{col}_masked"
            df[new_col] = df.apply(lambda r: mask_text(r[col], r), axis=1)

    return df

df = apply_masking(df, k_list=(2,3,4,5,6))

import re

def clean_reasoning_tags(text: str) -> str:
    """Remove <reasoning> and <r> tags, return joined plain text lines."""
    if not isinstance(text, str):
        return ""
    # extract <r>...</r>
    clauses = re.findall(r"<r>(.*?)</r>", text, flags=re.DOTALL|re.IGNORECASE)
    # join them as plain lines (or you can use " ".join if you prefer one paragraph)
    return "\n".join(c.strip() for c in clauses if c.strip())

def clean_clause_columns(df, k_list=(2,3,4,5,6)):
    for K in k_list:
        col = f"clauses_from_expl_k{K}_masked"
        if col in df.columns:
            df[col] = df[col].apply(clean_reasoning_tags)
    return df

df = clean_clause_columns(df, k_list=(2,3,4,5,6))


logger = logging.getLogger("RewriteLogger")
logger.setLevel(logging.INFO)  # or DEBUG for more detail
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

per_model_summaries = evaluate_models_by_expl_types(
    model_names=[("Qwen/Qwen3-1.7B","")],
    df=df,
    option_columns=["choice_A","choice_B","choice_C","choice_D"],
    option_letters=["A","B","C","D"],
    expl_variants=expl_variants,
    logger=logger
)

# Peek:
for short, summ in per_model_summaries.items():
    print(f"\n== Summary for {short} ==")
    print(summ.to_string(index=False))
