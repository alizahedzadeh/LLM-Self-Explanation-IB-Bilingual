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