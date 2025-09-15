CONCISE_REWRITE_PROMPT_TEMPLATE = """
You are an expert at rewriting long explanations into shorter, more concise ones — while preserving the reasoning and correctness.

Your task is to rewrite the explanation below in a more concise form. Keep the chain of reasoning intact, do not remove important logic steps, and ensure the answer can still be inferred from the rewritten explanation.

Original Explanation:
"{explanation}"

Now, rewrite the explanation to be more concise while keeping all essential reasoning:
"""

REFLECTION_PROMPT_TEMPLATE = """
Evaluate the following concise explanation for logical completeness and sufficiency. Was it enough to infer the correct answer? Suggest improvements if any.

Explanation:
"{explanation}"
"""


PREDICTION_BASE_PROMPT_TEMPLATE = '''
You are given a multiple-choice question. 

Step 1: Based on your knowledge and reasoning, select the most likely correct answer.  
Step 2: Justify your answer with clear reasoning and explanation.

Instructions:
- Use logical reasoning to determine the best answer.
- Do not reference the other answer options in your explanation.
- Keep the explanation concise but informative (2-4 sentences).
- Provide clear reasoning for your choice.

---

Question: {question}

Options:
A) {option_A}  
B) {option_B}  
C) {option_C}  
D) {option_D}

Respond in this format:

<prediction>[A/B/C/D]</prediction>  
<explanation>[Your reasoning and justification for the answer]</explanation>
'''




PREDICTION_BASE_PROMPT_TEMPLATE_PERSIAN = '''
You are given a multiple-choice question written in **Persian**.

Step 1: Based on your knowledge and reasoning, select the most likely correct answer.  
Step 2: Justify your answer with clear reasoning and explanation.

Instructions:
- Use logical reasoning to determine the best answer.
- Do not reference the other answer options in your explanation.
- Read the question and options carefully (they are in Persian).  
- Choose the most likely correct option (A, B, C, or D).  
- Keep the explanation concise but informative in **Persian language** (2-4 sentences).
- Provide clear reasoning for your choice.  

---
Question: {question}

Options:
A) {option_A}  
B) {option_B}  
C) {option_C}  
D) {option_D}

Respond in this format:

<prediction>[A/B/C/D]</prediction>
<explanation>[Your explanation in Persian]</explanation>
'''

# --- System prompt for our task ---
SYSTEM_PROMPT = """You are a reasoning assistant for multiple-choice QA.
Always respond in this exact format:

<prediction>[A/B/C/D]</prediction>
<explanation>[Your short explanation]</explanation>
"""
