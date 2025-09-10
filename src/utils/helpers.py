import re
import pandas as pd

# Explanation masking helpers
def mask_explanation(row):
    explanation = row["explanation"]
    correct_letter = row["actual_answer"]
    if pd.isna(explanation):
        return ""
    
    # Mask correct option letter
    explanation = re.sub(rf'\b{correct_letter}\b', '[MASK]', explanation, flags=re.IGNORECASE)

    # Mask all option texts
    for col in ['choice_A', 'choice_B', 'choice_C', 'choice_D']:
        option_text = re.escape(str(row[col]))
        explanation = re.sub(option_text, '[MASK]', explanation, flags=re.IGNORECASE)
    
    return explanation

def mask_explanation_limit(row):
    explanation = row["rewritten_explanation"]
    correct_letter = row["actual_answer"]
    if pd.isna(explanation):
        return ""
    
    # Mask correct option letter
    explanation = re.sub(rf'\b{correct_letter}\b', '[MASK]', explanation, flags=re.IGNORECASE)

    # Mask all option texts
    for col in ['choice_A', 'choice_B', 'choice_C', 'choice_D']:
        option_text = re.escape(str(row[col]))
        explanation = re.sub(option_text, '[MASK]', explanation, flags=re.IGNORECASE)
    
    return explanation

def mask_explanation_fa(row):
    explanation = row["explanation"]
    correct_letter = row["answerKey"]
    if pd.isna(explanation):
        return ""
    
    # Mask correct option letter
    explanation = re.sub(rf'\b{correct_letter}\b', '[MASK]', explanation, flags=re.IGNORECASE)

    # Mask all option texts
    for col in ['choice_A_fa', 'choice_B_fa', 'choice_C_fa', 'choice_D_fa']:
        option_text = re.escape(str(row[col]))
        explanation = re.sub(option_text, '[MASK]', explanation, flags=re.IGNORECASE)
    
    return explanation


def count_tokens(text, enc):
    return len(enc.encode(text))

def trim_to_token_limit(text, max_tokens, enc):
    tokens = enc.encode(text)
    trimmed = tokens[:max_tokens]
    return enc.decode(trimmed)