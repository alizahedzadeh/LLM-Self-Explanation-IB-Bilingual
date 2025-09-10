import gc
import torch
import json
import logging
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def evaluate_models_with_explanation(model_names, df, option_columns, option_letters, logger):
    """
    Evaluate a list of language models on a multiple-choice QA dataset
    with and without explanations, and compute explanation sufficiency.

    Args:
        model_names (list of tuples): List of (model_name, short_name)
        df (pd.DataFrame): The dataset containing questions, options, answerKey, and explanations
        option_columns (list): Column names for the multiple-choice options
        option_letters (list): Option labels (e.g. ["A", "B", "C", "D"])
        logger (logging.Logger): Logger instance
    """

    for model_name, short_name in model_names:
        logger.info(f"\n=== Running on model: {model_name} ===")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        model.eval()

        results = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Testing {short_name}"):
            question = str(row['question'])
            options = [str(row[col]) for col in option_columns]
            explanation = str(row['explanation']) if 'explanation' in row and pd.notna(row['explanation']) else ""
            gold_letter = row['answerKey']

            def score_options(prompt):
                logprobs = []
                logs = []
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
                    log_strs = []
                    total_log_prob = 0
                    for j, token_id in enumerate(target_ids):
                        log_prob = log_probs[j, token_id].item()
                        token_str = tokenizer.decode([token_id])
                        log_strs.append(f"Token: '{token_str.strip()}' | Log-prob: {log_prob:.4f}")
                        total_log_prob += log_prob
                    logprobs.append(total_log_prob)
                    logs.append("\n".join(log_strs))
                probs = torch.softmax(torch.tensor(logprobs, dtype=torch.float32), dim=0)
                pred_idx = probs.argmax().item()
                return option_letters[pred_idx], probs[pred_idx].item(), probs, logs

            # Prompt without explanation
            prompt_noexp = (
                question + "\nOptions:\n" +
                "\n".join([f"{l}) {t}" for l, t in zip(option_letters, options)]) +
                "\nThe answer is "
            )
            if not prompt_noexp.endswith(" "): prompt_noexp += " "
            pred_letter, pred_conf, probs, _ = score_options(prompt_noexp)

            # Prompt with explanation
            prompt_exp = (
                question + "\nExplanation: " + explanation +
                "\nOptions:\n" +
                "\n".join([f"{l}) {t}" for l, t in zip(option_letters, options)]) +
                "\nThe answer is "
            )
            if not prompt_exp.endswith(" "): prompt_exp += " "
            pred_letter_exp, pred_conf_exp, probs_exp, _ = score_options(prompt_exp)

            is_correct_noexp = (pred_letter == gold_letter)
            is_correct_exp = (pred_letter_exp == gold_letter)
            sufficiency = is_correct_exp

            usefulness = (
                (not is_correct_noexp and is_correct_exp) or
                (is_correct_noexp and is_correct_exp and (pred_conf_exp > pred_conf))
            )

            logger.info(
                f"[{short_name}] [Q{idx}] Gold: {gold_letter} | Pred_noexp: {pred_letter} ({pred_conf:.3f}) | "
                f"Pred_exp: {pred_letter_exp} ({pred_conf_exp:.3f}) | "
                f"Correct_noexp: {is_correct_noexp} | Correct_exp: {is_correct_exp} | Sufficient: {sufficiency}"
            )

            results.append({
                'index': row['row_id'] if 'row_id' in row else idx,
                'question': question,
                'options': options,
                'gold': gold_letter,
                'explanation': explanation,
                'pred_noexp': pred_letter,
                'prob_noexp': pred_conf,
                'pred_exp': pred_letter_exp,
                'prob_exp': pred_conf_exp,
                'is_correct_noexp': is_correct_noexp,
                'is_correct_exp': is_correct_exp,
                'sufficiency': sufficiency,
                'usefulness': usefulness,
                'probs_noexp': probs.cpu().numpy(),
                'probs_exp': probs_exp.cpu().numpy(),
            })

        # Save per-model results
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'results_{short_name}.csv', index=False)

        # Log metrics
        acc_noexp = results_df['is_correct_noexp'].mean()
        acc_exp = results_df['is_correct_exp'].mean()
        suff_rate = results_df['sufficiency'].mean()
        logger.info(f"\nModel: {short_name} | Acc noexp: {acc_noexp:.3f} | Acc exp: {acc_exp:.3f} | Sufficiency: {suff_rate:.3f}")

        # Cleanup
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    logger.info("All models tested.")
    return results_df


def evaluate_models_with_masked_explanation(model_names, df, option_columns, option_letters, logger):
    """
    Evaluate a list of language models on a multiple-choice QA dataset
    with and without masked explanations, and compute explanation sufficiency.

    Args:
        model_names (list of tuples): List of (model_name, short_name)
        df (pd.DataFrame): The dataset containing questions, options, answerKey, and masked_explanations
        option_columns (list): Column names for the multiple-choice options
        option_letters (list): Option labels (e.g. ["A", "B", "C", "D"])
        logger (logging.Logger): Logger instance
    """

    for model_name, short_name in model_names:
        logger.info(f"\n=== Running on model: {model_name} ===")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        model.eval()

        results = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Testing {short_name}"):
            question = str(row['question'])
            options = [str(row[col]) for col in option_columns]
            masked_explanation = str(row['masked_explanation']) if 'masked_explanation' in row and pd.notna(row['masked_explanation']) else ""
            gold_letter = row['actual_answer']

            def score_options(prompt):
                logprobs = []
                logs = []
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
                    log_strs = []
                    total_log_prob = 0
                    for j, token_id in enumerate(target_ids):
                        log_prob = log_probs[j, token_id].item()
                        token_str = tokenizer.decode([token_id])
                        log_strs.append(f"Token: '{token_str.strip()}' | Log-prob: {log_prob:.4f}")
                        total_log_prob += log_prob
                    logprobs.append(total_log_prob)
                    logs.append("\n".join(log_strs))
                probs = torch.softmax(torch.tensor(logprobs, dtype=torch.float32), dim=0)
                pred_idx = probs.argmax().item()
                return option_letters[pred_idx], probs[pred_idx].item(), probs, logs

            # Prompt without explanation
            prompt_noexp = (
                question + "\nOptions:\n" +
                "\n".join([f"{l}) {t}" for l, t in zip(option_letters, options)]) +
                "\nThe answer is "
            )
            if not prompt_noexp.endswith(" "): prompt_noexp += " "
            pred_letter, pred_conf, probs, _ = score_options(prompt_noexp)

            # Prompt with masked explanation
            prompt_exp = (
                question + "\nExplanation: " + masked_explanation +
                "\nOptions:\n" +
                "\n".join([f"{l}) {t}" for l, t in zip(option_letters, options)]) +
                "\nThe answer is "
            )
            if not prompt_exp.endswith(" "): prompt_exp += " "
            pred_letter_exp, pred_conf_exp, probs_exp, _ = score_options(prompt_exp)

            is_correct_noexp = (pred_letter == gold_letter)
            is_correct_exp = (pred_letter_exp == gold_letter)
            sufficiency = is_correct_exp

            usefulness = (
                (not is_correct_noexp and is_correct_exp) or
                (is_correct_noexp and is_correct_exp and (pred_conf_exp > pred_conf))
            )


            logger.info(
                f"[{short_name}] [Q{idx}] Gold: {gold_letter} | Pred_noexp: {pred_letter} ({pred_conf:.3f}) | "
                f"Pred_exp: {pred_letter_exp} ({pred_conf_exp:.3f}) | "
                f"Correct_noexp: {is_correct_noexp} | Correct_exp: {is_correct_exp} | Sufficient: {sufficiency}"
            )

            results.append({
                'index': row['row_id'] if 'row_id' in row else idx,
                'question': question,
                'options': options,
                'gold': gold_letter,
                'masked_explanation': masked_explanation,
                'pred_noexp': pred_letter,
                'prob_noexp': pred_conf,
                'pred_exp': pred_letter_exp,
                'prob_exp': pred_conf_exp,
                'is_correct_noexp': is_correct_noexp,
                'is_correct_exp': is_correct_exp,
                'sufficiency': sufficiency,
                'usefulness': usefulness,
                'probs_noexp': probs.cpu().numpy(),
                'probs_exp': probs_exp.cpu().numpy(),
            })

        # Save per-model results
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'results_{short_name}_masked.csv', index=False)

        # Log metrics
        acc_noexp = results_df['is_correct_noexp'].mean()
        acc_exp = results_df['is_correct_exp'].mean()
        suff_rate = results_df['sufficiency'].mean()
        logger.info(f"\nModel: {short_name} | Acc noexp: {acc_noexp:.3f} | Acc exp: {acc_exp:.3f} | Sufficiency: {suff_rate:.3f}")

        # Cleanup
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    logger.info("All models tested with masked explanations.")
    return results_df

def evaluate_models_with_explanation_fa(model_names, df, option_columns, option_letters, logger):
    """
    Evaluate a list of language models on a multiple-choice QA dataset
    with and without explanations, and compute explanation sufficiency.

    Args:
        model_names (list of tuples): List of (model_name, short_name)
        df (pd.DataFrame): The dataset containing questions, options, answerKey, and explanations
        option_columns (list): Column names for the multiple-choice options
        option_letters (list): Option labels (e.g. ["A", "B", "C", "D"])
        logger (logging.Logger): Logger instance
    """

    for model_name, short_name in model_names:
        logger.info(f"\n=== Running on model: {model_name} ===")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        model.eval()

        results = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Testing {short_name}"):
            question = str(row['question_fa'])
            options = [str(row[col]) for col in option_columns]
            explanation = str(row['explanation']) if 'explanation' in row and pd.notna(row['explanation']) else ""
            gold_letter = row['answerKey']

            def score_options(prompt):
                logprobs = []
                logs = []
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
                    log_strs = []
                    total_log_prob = 0
                    for j, token_id in enumerate(target_ids):
                        log_prob = log_probs[j, token_id].item()
                        token_str = tokenizer.decode([token_id])
                        log_strs.append(f"Token: '{token_str.strip()}' | Log-prob: {log_prob:.4f}")
                        total_log_prob += log_prob
                    logprobs.append(total_log_prob)
                    logs.append("\n".join(log_strs))
                probs = torch.softmax(torch.tensor(logprobs, dtype=torch.float32), dim=0)
                pred_idx = probs.argmax().item()
                return option_letters[pred_idx], probs[pred_idx].item(), probs, logs

            # Prompt without explanation
            prompt_noexp = (
                question + "\nOptions:\n" +
                "\n".join([f"{l}) {t}" for l, t in zip(option_letters, options)]) +
                "\nThe answer is "
            )
            if not prompt_noexp.endswith(" "): prompt_noexp += " "
            pred_letter, pred_conf, probs, _ = score_options(prompt_noexp)

            # Prompt with explanation
            prompt_exp = (
                question + "\nExplanation: " + explanation +
                "\nOptions:\n" +
                "\n".join([f"{l}) {t}" for l, t in zip(option_letters, options)]) +
                "\nThe answer is "
            )
            if not prompt_exp.endswith(" "): prompt_exp += " "
            pred_letter_exp, pred_conf_exp, probs_exp, _ = score_options(prompt_exp)

            is_correct_noexp = (pred_letter == gold_letter)
            is_correct_exp = (pred_letter_exp == gold_letter)
            sufficiency = is_correct_exp
            usefulness = (
                (not is_correct_noexp and is_correct_exp) or
                (is_correct_noexp and is_correct_exp and (pred_conf_exp > pred_conf))
            )

            logger.info(
                f"[{short_name}] [Q{idx}] Gold: {gold_letter} | Pred_noexp: {pred_letter} ({pred_conf:.3f}) | "
                f"Pred_exp: {pred_letter_exp} ({pred_conf_exp:.3f}) | "
                f"Correct_noexp: {is_correct_noexp} | Correct_exp: {is_correct_exp} | Sufficient: {sufficiency}"
            )

            results.append({
                'index': row['row_id'] if 'row_id' in row else idx,
                'question': question,
                'options': options,
                'gold': gold_letter,
                'explanation': explanation,
                'pred_noexp': pred_letter,
                'prob_noexp': pred_conf,
                'pred_exp': pred_letter_exp,
                'prob_exp': pred_conf_exp,
                'is_correct_noexp': is_correct_noexp,
                'is_correct_exp': is_correct_exp,
                'sufficiency': sufficiency,
                'usefulness': usefulness,
                'probs_noexp': probs.cpu().numpy(),
                'probs_exp': probs_exp.cpu().numpy(),
            })

        # Save per-model results
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'results_{short_name}.csv', index=False)

        # Log metrics
        acc_noexp = results_df['is_correct_noexp'].mean()
        acc_exp = results_df['is_correct_exp'].mean()
        suff_rate = results_df['sufficiency'].mean()
        logger.info(f"\nModel: {short_name} | Acc noexp: {acc_noexp:.3f} | Acc exp: {acc_exp:.3f} | Sufficiency: {suff_rate:.3f}")

        # Cleanup
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    logger.info("All models tested.")
    return results_df


def evaluate_models_with_masked_explanation_fa(model_names, df, option_columns, option_letters, logger):
    """
    Evaluate a list of language models on a multiple-choice QA dataset
    with and without masked explanations, and compute explanation sufficiency.

    Args:
        model_names (list of tuples): List of (model_name, short_name)
        df (pd.DataFrame): The dataset containing questions, options, answerKey, and masked_explanations
        option_columns (list): Column names for the multiple-choice options
        option_letters (list): Option labels (e.g. ["A", "B", "C", "D"])
        logger (logging.Logger): Logger instance
    """

    for model_name, short_name in model_names:
        logger.info(f"\n=== Running on model: {model_name} ===")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        model.eval()

        results = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Testing {short_name}"):
            question = str(row['question_fa'])
            options = [str(row[col]) for col in option_columns]
            masked_explanation = str(row['masked_explanation']) if 'masked_explanation' in row and pd.notna(row['masked_explanation']) else ""
            gold_letter = row['answerKey']

            def score_options(prompt):
                logprobs = []
                logs = []
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
                    log_strs = []
                    total_log_prob = 0
                    for j, token_id in enumerate(target_ids):
                        log_prob = log_probs[j, token_id].item()
                        token_str = tokenizer.decode([token_id])
                        log_strs.append(f"Token: '{token_str.strip()}' | Log-prob: {log_prob:.4f}")
                        total_log_prob += log_prob
                    logprobs.append(total_log_prob)
                    logs.append("\n".join(log_strs))
                probs = torch.softmax(torch.tensor(logprobs, dtype=torch.float32), dim=0)
                pred_idx = probs.argmax().item()
                return option_letters[pred_idx], probs[pred_idx].item(), probs, logs

            # Prompt without explanation
            prompt_noexp = (
                question + "\nOptions:\n" +
                "\n".join([f"{l}) {t}" for l, t in zip(option_letters, options)]) +
                "\nThe answer is "
            )
            if not prompt_noexp.endswith(" "): prompt_noexp += " "
            pred_letter, pred_conf, probs, _ = score_options(prompt_noexp)

            # Prompt with masked explanation
            prompt_exp = (
                question + "\nExplanation: " + masked_explanation +
                "\nOptions:\n" +
                "\n".join([f"{l}) {t}" for l, t in zip(option_letters, options)]) +
                "\nThe answer is "
            )
            if not prompt_exp.endswith(" "): prompt_exp += " "
            pred_letter_exp, pred_conf_exp, probs_exp, _ = score_options(prompt_exp)

            is_correct_noexp = (pred_letter == gold_letter)
            is_correct_exp = (pred_letter_exp == gold_letter)
            sufficiency = is_correct_exp

            usefulness = (
                (not is_correct_noexp and is_correct_exp) or
                (is_correct_noexp and is_correct_exp and (pred_conf_exp > pred_conf))
            )

            logger.info(
                f"[{short_name}] [Q{idx}] Gold: {gold_letter} | Pred_noexp: {pred_letter} ({pred_conf:.3f}) | "
                f"Pred_exp: {pred_letter_exp} ({pred_conf_exp:.3f}) | "
                f"Correct_noexp: {is_correct_noexp} | Correct_exp: {is_correct_exp} | Sufficient: {sufficiency}"
            )

            results.append({
                'index': row['row_id'] if 'row_id' in row else idx,
                'question': question,
                'options': options,
                'gold': gold_letter,
                'masked_explanation': masked_explanation,
                'pred_noexp': pred_letter,
                'prob_noexp': pred_conf,
                'pred_exp': pred_letter_exp,
                'prob_exp': pred_conf_exp,
                'is_correct_noexp': is_correct_noexp,
                'is_correct_exp': is_correct_exp,
                'sufficiency': sufficiency,
                'usefulness': usefulness,
                'probs_noexp': probs.cpu().numpy(),
                'probs_exp': probs_exp.cpu().numpy(),
            })

        # Save per-model results
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'results_{short_name}_masked.csv', index=False)

        # Log metrics
        acc_noexp = results_df['is_correct_noexp'].mean()
        acc_exp = results_df['is_correct_exp'].mean()
        suff_rate = results_df['sufficiency'].mean()
        logger.info(f"\nModel: {short_name} | Acc noexp: {acc_noexp:.3f} | Acc exp: {acc_exp:.3f} | Sufficiency: {suff_rate:.3f}")

        # Cleanup
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    logger.info("All models tested with masked explanations.")

    return results_df







'''
OPTIMIZED VERSION OF EVALUATORS
'''



def evaluate_models_with_explanation_optimized(
        model_names,                   # [(model_id, short_name), ...]
        df,                            # DataFrame with columns: question, answerKey, explanation, plus option cols
        option_columns,                # e.g. ["choice_A", "choice_B", "choice_C", "choice_D"]
        option_letters,                # ["A", "B", "C", "D"]
        logger: logging.Logger,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
    """
    Evaluate each (model_name, short_name) on the dataset in two modes:
      • without explanation
      • with explanation
    Returns: list of dicts ready for CSV.
    """

    all_results = []

    for model_name, short_name in model_names:
        logger.info(f"\n=== Evaluating model: {model_name} ===")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto" if device.startswith("cuda") else None
        ).to(device).eval()

        def score_options(prompt: str):
            """Return (pred_letter, pred_conf, probs_list)."""
            # 1) encode prompt once
            base_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)[0]

            logprobs = []
            for letter in option_letters:
                # prepend space so tokeniser keeps it separate
                tgt_ids = tokenizer(" " + letter, add_special_tokens=False,
                                    return_tensors="pt").input_ids.to(device)[0]
                full_ids = torch.cat([base_ids, tgt_ids])

                # 2) forward pass – only need logits at last position before target
                with torch.inference_mode():
                    logits = model(full_ids.unsqueeze(0)).logits  # [1, L, V]

                # last position before first target token
                last_log = logits[0, -tgt_ids.size(0)-1, :]
                lp = F.log_softmax(last_log, dim=-1)[tgt_ids[0]].item()
                logprobs.append(lp)

            probs = torch.softmax(torch.tensor(logprobs), dim=0)
            pred_idx = int(probs.argmax())
            return option_letters[pred_idx], float(probs[pred_idx]), probs.tolist()

        for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"Testing {short_name}"):
            q_text    = str(row.question)
            gold      = row.answerKey
            expl      = getattr(row, "explanation", "") or ""
            options   = [str(getattr(row, col)) for col in option_columns]

            # ----------- prompt (no explanation) -----------
            prompt_noexp = (
                q_text + "\nOptions:\n" +
                "\n".join(f"{l}) {t}" for l, t in zip(option_letters, options)) +
                "\nThe answer is "
            )
            pred_noexp, conf_noexp, probs_noexp = score_options(prompt_noexp)

            # ----------- prompt (with explanation) ----------
            prompt_exp = (
                q_text + f"\nExplanation: {expl}\nOptions:\n" +
                "\n".join(f"{l}) {t}" for l, t in zip(option_letters, options)) +
                "\nThe answer is "
            )
            pred_exp, conf_exp, probs_exp = score_options(prompt_exp)

            # ----------- metrics ----------------------------
            is_correct_noexp = (pred_noexp == gold)
            is_correct_exp   = (pred_exp   == gold)

            # Sufficiency: توضیح کافیست ⇔ پاسخِ با توضیح درست است
            sufficiency = is_correct_exp

            # Usefulness: توضیح عملکرد را بهبود داده یا اعتماد را افزایش داده
            usefulness = (
                (not is_correct_noexp and is_correct_exp) or
                (is_correct_noexp and is_correct_exp and conf_exp > conf_noexp)
            )

            logger.debug(
                f"[{short_name}] Gold={gold} | "
                f"noExp={pred_noexp}({conf_noexp:.3f}) "
                f"exp={pred_exp}({conf_exp:.3f}) | "
                f"Suff={sufficiency} | Useful={usefulness}"
            )

            all_results.append({
                "model": short_name,
                "question": q_text,
                "options": options,
                "gold": gold,
                "explanation": expl,
                "pred_noexp": pred_noexp,
                "prob_noexp": conf_noexp,
                "pred_exp": pred_exp,
                "prob_exp": conf_exp,
                "is_correct_noexp": is_correct_noexp,
                "is_correct_exp": is_correct_exp,
                "sufficiency": sufficiency,
                "usefulness": usefulness,
                "probs_noexp": json.dumps(probs_noexp),
                "probs_exp": json.dumps(probs_exp)
            })

        # optional: free GPU memory per-model
        del model
        torch.cuda.empty_cache()

    return all_results



def evaluate_models_with_masked_explanation_optimized(
        model_names,         # [(model_id, short_name), …]
        df,                  # DataFrame → question, answerKey, masked_explanation, options
        option_columns,      # e.g. ["choice_A", "choice_B", …]
        option_letters,      # ["A","B","C","D"]
        logger: logging.Logger,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
    """
    Run each model twice on every sample:
      • without explanation
      • with *masked* explanation (answer letters removed)
    Returns a DataFrame of all results and also saves per-model CSVs.
    """

    all_results = []

    for model_name, short_name in model_names:
        logger.info(f"\n=== Evaluating {short_name} ({model_name}) with MASKED explanations ===")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto" if device.startswith("cuda") else None
        ).to(device).eval()

        def score_options(prompt: str):
            """Single-token scoring; returns (pred_letter, pred_conf, probs_list)."""
            prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)[0]
            logprobs = []
            for letter in option_letters:
                tgt_ids = tokenizer(" " + letter, add_special_tokens=False,
                                    return_tensors="pt").input_ids.to(device)[0]
                full_ids = torch.cat([prompt_ids, tgt_ids])

                with torch.inference_mode():
                    logits = model(full_ids.unsqueeze(0)).logits

                last_log = logits[0, -tgt_ids.size(0)-1, :]          # token before letter
                lp = F.log_softmax(last_log, dim=-1)[tgt_ids[0]].item()
                logprobs.append(lp)

            probs = torch.softmax(torch.tensor(logprobs), dim=0)
            idx = int(probs.argmax())
            return option_letters[idx], float(probs[idx]), probs.tolist()

        per_model = []  # collect rows for this model

        for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"Testing {short_name}"):
            q_text   = str(row.question)
            gold     = row.answerKey
            m_expl   = getattr(row, "masked_explanation", "") or ""
            options  = [str(getattr(row, col)) for col in option_columns]

            # -------- prompt: no explanation ----------
            prompt_noexp = (
                q_text + "\nOptions:\n" +
                "\n".join(f"{l}) {t}" for l, t in zip(option_letters, options)) +
                "\nThe answer is "
            )
            pred_noexp, conf_noexp, probs_noexp = score_options(prompt_noexp)

            # -------- prompt: with MASKED explanation ---
            prompt_masked = (
                q_text + f"\nExplanation: {m_expl}\nOptions:\n" +
                "\n".join(f"{l}) {t}" for l, t in zip(option_letters, options)) +
                "\nThe answer is "
            )
            pred_exp, conf_exp, probs_exp = score_options(prompt_masked)

            is_correct_noexp = (pred_noexp == gold)
            is_correct_exp   = (pred_exp   == gold)

            # sufficiency: آیا مدل با توضیح (حتی ماسک‌شده) درست گفت؟
            sufficiency = is_correct_exp

            # usefulness: بهبود از خطا به درست، یا افزایش اعتماد پاسخ درست
            usefulness = (
                (not is_correct_noexp and is_correct_exp) or
                (is_correct_noexp and is_correct_exp and conf_exp > conf_noexp)
            )

            per_model.append({
                "model": short_name,
                "question": q_text,
                "options": options,
                "gold": gold,
                "masked_explanation": m_expl,
                "pred_noexp": pred_noexp,
                "prob_noexp": conf_noexp,
                "pred_exp": pred_exp,
                "prob_exp": conf_exp,
                "is_correct_noexp": is_correct_noexp,
                "is_correct_exp": is_correct_exp,
                "sufficiency": sufficiency,
                "usefulness": usefulness,
                "probs_noexp": json.dumps(probs_noexp),
                "probs_exp": json.dumps(probs_exp)
            })

        # ---------- save + log per-model -----------
        per_df = pd.DataFrame(per_model)
        per_df.to_csv(f"results_{short_name}_masked.csv", index=False)

        acc0, acc1 = per_df["is_correct_noexp"].mean(), per_df["is_correct_exp"].mean()
        logger.info(f"Model {short_name} — Acc(no-exp)={acc0:.3f} | "
                    f"Acc(masked)={acc1:.3f} | Suff={per_df['sufficiency'].mean():.3f}")

        all_results.extend(per_model)

        # cleanup GPU
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    logger.info("✅ Finished all masked-explanation evaluations.")
    return pd.DataFrame(all_results)





# ──────────────────────────────────────
# 1) با توضیح (فارسی)
# ──────────────────────────────────────
def evaluate_models_with_explanation_fa_optimized(
        model_names, df, option_columns, option_letters, logger,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"):

    all_rows = []

    for model_id, short_name in model_names:
        logger.info(f"\n=== {short_name}: evaluating WITH explanations (FA) ===")
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto",
            device_map="auto" if device.startswith("cuda") else None
        ).to(device).eval()

        def score(prompt: str):
            base_ids = tok(prompt, return_tensors="pt").input_ids.to(device)[0]
            lp = []
            for L in option_letters:
                tgt_ids = tok(" " + L, add_special_tokens=False,
                              return_tensors="pt").input_ids.to(device)[0]
                ids = torch.cat([base_ids, tgt_ids])
                with torch.inference_mode():
                    logits = model(ids.unsqueeze(0)).logits
                last = logits[0, -tgt_ids.size(0)-1, :]
                lp.append(F.log_softmax(last, dim=-1)[tgt_ids[0]].item())
            probs = torch.softmax(torch.tensor(lp), dim=0)
            idx = int(probs.argmax())
            return option_letters[idx], float(probs[idx]), probs.tolist()

        per_model = []
        for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"{short_name}-fa"):
            q = str(row.question_fa)
            expl = getattr(row, "explanation", "") or ""
            gold = row["actual_answer"]
            opts = [str(getattr(row, col)) for col in option_columns]

            # بدون توضیح
            p0 = f"{q}\nOptions:\n" + "\n".join(f"{l}) {t}" for l,t in zip(option_letters, opts)) + "\nThe answer is "
            pred0, conf0, probs0 = score(p0)

            # با توضیح
            p1 = f"{q}\nExplanation: {expl}\nOptions:\n" + \
                 "\n".join(f"{l}) {t}" for l,t in zip(option_letters, opts)) + "\nThe answer is "
            pred1, conf1, probs1 = score(p1)

            correct0, correct1 = pred0==gold, pred1==gold
            suff  = correct1
            usefu = ((not correct0 and correct1) or
                     (correct0 and correct1 and conf1>conf0))

            per_model.append({
                "model": short_name,
                "question_fa": q,
                "options": opts,
                "gold": gold,
                "explanation": expl,
                "pred_noexp": pred0,  "prob_noexp": conf0,
                "pred_exp":  pred1,  "prob_exp":  conf1,
                "is_correct_noexp": correct0,
                "is_correct_exp":   correct1,
                "sufficiency": suff,
                "usefulness": usefu,
                "probs_noexp": json.dumps(probs0),
                "probs_exp":  json.dumps(probs1)
            })

        df_model = pd.DataFrame(per_model)
        df_model.to_csv(f"results_{short_name}.csv", index=False)
        logger.info(f"{short_name}: Acc0={df_model.is_correct_noexp.mean():.3f} | "
                    f"Acc1={df_model.is_correct_exp.mean():.3f} | Suff={df_model.sufficiency.mean():.3f}")
        all_rows.extend(per_model)

        del model, tok
        torch.cuda.empty_cache(); gc.collect()

    return pd.DataFrame(all_rows)

# ──────────────────────────────────────
# 2) با توضیحِ ماسک‌شده (فارسی)
# ──────────────────────────────────────
def evaluate_models_with_masked_explanation_fa_optimized(
        model_names, df, option_columns, option_letters, logger,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"):

    all_rows = []

    for model_id, short_name in model_names:
        logger.info(f"\n=== {short_name}: evaluating MASKED explanations (FA) ===")
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto",
            device_map="auto" if device.startswith("cuda") else None
        ).to(device).eval()

        def score(prompt: str):
            base_ids = tok(prompt, return_tensors="pt").input_ids.to(device)[0]
            lp = []
            for L in option_letters:
                tgt_ids = tok(" " + L, add_special_tokens=False,
                              return_tensors="pt").input_ids.to(device)[0]
                ids = torch.cat([base_ids, tgt_ids])
                with torch.inference_mode():
                    logits = model(ids.unsqueeze(0)).logits
                last = logits[0, -tgt_ids.size(0)-1, :]
                lp.append(F.log_softmax(last, dim=-1)[tgt_ids[0]].item())
            probs = torch.softmax(torch.tensor(lp), dim=0)
            idx = int(probs.argmax())
            return option_letters[idx], float(probs[idx]), probs.tolist()

        per_model = []
        for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"{short_name}-fa-mask"):
            q = str(row.question_fa)
            expl = getattr(row, "masked_explanation", "") or ""
            gold= row.actual_answer
            opts= [str(getattr(row, col)) for col in option_columns]

            p0 = f"{q}\nOptions:\n" + "\n".join(f"{l}) {t}" for l,t in zip(option_letters, opts)) + "\nThe answer is "
            pred0, conf0, probs0 = score(p0)

            p1 = f"{q}\nExplanation: {expl}\nOptions:\n" + \
                 "\n".join(f"{l}) {t}" for l,t in zip(option_letters, opts)) + "\nThe answer is "
            pred1, conf1, probs1 = score(p1)

            correct0, correct1 = pred0==gold, pred1==gold
            suff  = correct1
            usefu = ((not correct0 and correct1) or
                     (correct0 and correct1 and conf1>conf0))

            per_model.append({
                "model": short_name,
                "question_fa": q,
                "options": opts,
                "gold": gold,
                "masked_explanation": expl,
                "pred_noexp": pred0,  "prob_noexp": conf0,
                "pred_exp":  pred1,  "prob_exp":  conf1,
                "is_correct_noexp": correct0,
                "is_correct_exp":   correct1,
                "sufficiency": suff,
                "usefulness": usefu,
                "probs_noexp": json.dumps(probs0),
                "probs_exp":  json.dumps(probs1)
            })

        df_model = pd.DataFrame(per_model)
        df_model.to_csv(f"results_{short_name}_masked.csv", index=False)
        logger.info(f"{short_name}: Acc0={df_model.is_correct_noexp.mean():.3f} | "
                    f"Acc1={df_model.is_correct_exp.mean():.3f} | Suff={df_model.sufficiency.mean():.3f}")
        all_rows.extend(per_model)

        del model, tok
        torch.cuda.empty_cache(); gc.collect()

    logger.info("✅ همهٔ مدل‌ها با توضیح و توضیح‌ِماسک‌شده (FA) ارزیابی شدند.")
    return pd.DataFrame(all_rows)
