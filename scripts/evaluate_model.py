import gc
import torch
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
            sufficiency = (not is_correct_noexp and is_correct_exp) or (is_correct_noexp and (pred_conf_exp > pred_conf))

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
        conf_rate = results_df['pred_conf_exp'].mean()
        logger.info(f"\nModel: {short_name} | Acc noexp: {acc_noexp:.3f} | Acc exp: {acc_exp:.3f} | Sufficiency: {suff_rate:.3f} | Confidence Rate: {conf_rate:.3f}")

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
            sufficiency = (not is_correct_noexp and is_correct_exp) or (is_correct_noexp and (pred_conf_exp > pred_conf))

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
        conf_rate = results_df['pred_conf_exp'].mean()
        logger.info(f"\nModel: {short_name} | Acc noexp: {acc_noexp:.3f} | Acc exp: {acc_exp:.3f} | Sufficiency: {suff_rate:.3f} | Confidence Rate: {conf_rate:.3f}")
        

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
            sufficiency = (not is_correct_noexp and is_correct_exp) or (is_correct_noexp and (pred_conf_exp > pred_conf))

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
            sufficiency = (not is_correct_noexp and is_correct_exp) or (is_correct_noexp and (pred_conf_exp > pred_conf))

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