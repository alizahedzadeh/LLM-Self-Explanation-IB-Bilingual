from typing import List, Dict, Optional, Union, Any, Tuple
import numpy as np
from model_utils import LLMClient
from data_structures import MCQAInstance, Explanation

class ExplanationEvaluator:
    """Evaluator for MCQA explanations"""
    
    def __init__(self, judge_client: LLMClient):
        self.judge_client = judge_client
    
    def evaluate_sufficiency(
        self, 
        mcqa: MCQAInstance, 
        explanation: Explanation, 
        get_prob: bool = True
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate the sufficiency of an explanation using a judge LLM
        Returns: (sufficiency_score, option_probs_dict)
        """
        options_text = "\n".join([f"{key}. {value}" for key, value in mcqa.options.items()])
        
        prompt = f"""
        Question: {mcqa.question}
        
        Options:
        {options_text}
        
        Explanation justifying one option: 
        {explanation.text}
        
        Based solely on the provided explanation, the correct option is:
        """
        
        # If we want just accuracy (not probabilities), use a standard completion
        if not get_prob:
            response = self.judge_client.generate(
                prompt, 
                temperature=0.1, 
                max_tokens=5
            )
            prediction = response["text"].strip().split()[0]  # Extract just the option letter
            
            # Clean up prediction (remove punctuation)
            prediction = prediction.rstrip('.,;:!?')
            
            # Calculate accuracy (1.0 if correct, 0.0 if not)
            sufficiency = 1.0 if prediction == explanation.chosen_option else 0.0
            option_probs = {option: 1.0 if option == prediction else 0.0 for option in mcqa.options.keys()}
            
            return sufficiency, option_probs
        
        # For probability-based evaluation, we need to get token probabilities
        # NOTE: This implementation is simplified and may need adjustment
        # based on the specific API capabilities of GPT-4o-mini
        response = self.judge_client.generate(
            prompt, 
            temperature=0.1, 
            max_tokens=5,
            get_logprobs=True
        )
        
        # Process logprobs to get option probabilities
        # This is a placeholder - actual implementation would depend on
        # how the API returns logprobs
        if "logprobs" in response:
            logprobs = response.get("logprobs", {})
            # Process logprobs to extract probabilities for each option
            # This is highly dependent on the specific API
            option_probs = {option: 0.0 for option in mcqa.options.keys()}
            
            # Example processing (would need to be adapted to actual API response format)
            for option in option_probs.keys():
                # Look for option in top logprobs
                option_probs[option] = np.exp(logprobs.get(option, -100))  # Default to very low probability
            
            # Normalize
            total = sum(option_probs.values())
            if total > 0:
                option_probs = {k: v/total for k, v in option_probs.items()}
            
            # Sufficiency is the probability of the correct option
            sufficiency = option_probs.get(explanation.chosen_option, 0.0)
        else:
            # Fallback to accuracy-based evaluation if logprobs not available
            prediction = response["text"].strip().split()[0]
            prediction = prediction.rstrip('.,;:!?')
            sufficiency = 1.0 if prediction == explanation.chosen_option else 0.0
            option_probs = {option: 1.0 if option == prediction else 0.0 for option in mcqa.options.keys()}
        
        return sufficiency, option_probs
    
    def evaluate_conciseness(self, explanation: Explanation) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate the conciseness of an explanation
        Returns: (conciseness_score, details)
        """
        # Basic conciseness is inverse of token count (normalized)
        # Lower token count → higher conciseness
        token_count = explanation.token_count
        
        # Secondary measure: Use judge LLM to rate redundancy
        prompt = f"""
        Evaluate the following explanation for conciseness:
        
        "{explanation.text}"
        
        Does this explanation contain unnecessary information or redundant arguments that aren't needed to justify the answer?
        Please rate the conciseness on a scale of 1 to 10, where:
        - 1 means extremely verbose with lots of redundancy and irrelevant information
        - 10 means perfectly concise with only the necessary information
        
        Answer with just a number from 1-10.
        """
        
        response = self.judge_client.generate(prompt, temperature=0.1, max_tokens=5)
        redundancy_rating = response["text"].strip()
        
        # Extract numeric rating
        try:
            rating = float(re.search(r'\d+', redundancy_rating).group())
            # Normalize to 0-1 scale
            normalized_rating = rating / 10.0
        except (AttributeError, ValueError):
            normalized_rating = 0.5  # Default if parsing fails
        
        # Calculate token-based conciseness
        # Use a sigmoid function to map token count to a 0-1 score
        # Parameters can be tuned
        max_reasonable_tokens = 300  # Adjust based on your data
        token_conciseness = 1.0 / (1.0 + np.exp((token_count - max_reasonable_tokens/2) / (max_reasonable_tokens/10)))
        
        # Combine measures (equal weights by default)
        combined_conciseness = 0.5 * token_conciseness + 0.5 * normalized_rating
        
        details = {
            "token_count": token_count,
            "token_conciseness": token_conciseness,
            "redundancy_rating": rating if 'rating' in locals() else None,
            "normalized_rating": normalized_rating
        }
        
        return combined_conciseness, details
    
    def evaluate_explanation(self, mcqa: MCQAInstance, explanation: Explanation) -> Explanation:
        """Evaluate both sufficiency and conciseness"""
        # Evaluate sufficiency
        sufficiency, option_probs = self.evaluate_sufficiency(mcqa, explanation)
        explanation.sufficiency_score = sufficiency
        
        # Evaluate conciseness
        conciseness, conciseness_details = self.evaluate_conciseness(explanation)
        explanation.conciseness_score = conciseness
        
        # Ensure token count is set
        if explanation.token_count is None:
            explanation.token_count = self.judge_client.count_tokens(explanation.text)
        
        return explanation