from typing import List, Dict, Optional, Union, Any, Tuple
import re
import numpy as np
from model_utils import LLMClient
from data_structures import MCQAInstance, Explanation
from explanation_evaluation import ExplanationEvaluator

class ExplanationPruner:
    """Class for pruning explanations to find the minimal sufficient subset"""
    
    def __init__(self, judge_client: LLMClient, evaluator: ExplanationEvaluator, sufficiency_threshold: float = 0.7):
        self.judge_client = judge_client
        self.evaluator = evaluator
        self.sufficiency_threshold = sufficiency_threshold
    
    def split_explanation_into_segments(self, explanation_text: str) -> List[str]:
        """Split an explanation into segments (sentences or logical units)"""
        # Simple split by sentences as a starting point
        segments = re.split(r'(?<=[.!?])\s+', explanation_text.strip())
        return [seg.strip() for seg in segments if seg.strip()]
    
    def prune_explanation(
        self, 
        mcqa: MCQAInstance, 
        explanation: Explanation
    ) -> Tuple[Explanation, Dict[str, Any]]:
        """
        Prune an explanation to find a minimal sufficient subset
        Returns: (pruned_explanation, pruning_details)
        """
        # Split explanation into segments
        segments = self.split_explanation_into_segments(explanation.text)
        
        # Short explanation - nothing to prune
        if len(segments) <= 2:
            return explanation, {"prunable": False, "original_segments": segments}
        
        # Evaluate original sufficiency
        original_sufficiency, _ = self.evaluator.evaluate_sufficiency(mcqa, explanation)
        
        # If original explanation isn't sufficiently sufficient, return as is
        if original_sufficiency < self.sufficiency_threshold:
            return explanation, {
                "prunable": False, 
                "reason": "original_insufficiency",
                "original_sufficiency": original_sufficiency
            }
        
        # Initialize tracking for pruning process
        remaining_indices = list(range(len(segments)))
        removed_indices = []
        current_sufficiency = original_sufficiency
        pruning_history = [{"indices": remaining_indices.copy(), "sufficiency": current_sufficiency}]
        
        # Try removing segments one by one
        while len(remaining_indices) > 1 and current_sufficiency >= self.sufficiency_threshold:
            best_removal_index = None
            best_removal_sufficiency = 0
            
            # Try removing each remaining segment
            for i, idx in enumerate(remaining_indices):
                test_indices = remaining_indices.copy()
                test_indices.pop(i)
                
                # Create pruned explanation
                test_text = " ".join([segments[idx] for idx in test_indices])
                test_explanation = Explanation(
                    text=test_text,
                    mcqa_id=explanation.mcqa_id,
                    chosen_option=explanation.chosen_option,
                    generation_method=f"{explanation.generation_method}_pruned"
                )
                
                # Evaluate sufficiency
                test_sufficiency, _ = self.evaluator.evaluate_sufficiency(mcqa, test_explanation)
                
                # If still sufficient enough and best so far, track it
                if test_sufficiency >= self.sufficiency_threshold and test_sufficiency > best_removal_sufficiency:
                    best_removal_index = i
                    best_removal_sufficiency = test_sufficiency
            
            # If we found a segment to remove
            if best_removal_index is not None:
                removed_idx = remaining_indices.pop(best_removal_index)
                removed_indices.append(removed_idx)
                current_sufficiency = best_removal_sufficiency
                
                # Track history
                pruning_history.append({
                    "indices": remaining_indices.copy(), 
                    "sufficiency": current_sufficiency,
                    "removed_segment": segments[removed_idx]
                })
            else:
                # No more segments can be removed while maintaining sufficiency
                break
        
        # Create final pruned explanation
        pruned_text = " ".join([segments[idx] for idx in remaining_indices])
        pruned_explanation = Explanation(
            text=pruned_text,
            mcqa_id=explanation.mcqa_id,
            chosen_option=explanation.chosen_option,
            generation_method=f"{explanation.generation_method}_bottleneck"
        )
        
        # Update token count
        pruned_explanation.token_count = self.judge_client.count_tokens(pruned_text)
        
        # Calculate reduction metrics
        original_tokens = explanation.token_count or self.judge_client.count_tokens(explanation.text)
        token_reduction = 1 - (pruned_explanation.token_count / original_tokens)
        segment_reduction = 1 - (len(remaining_indices) / len(segments))
        
        # Compile details
        pruning_details = {
            "prunable": True,
            "original_segments": segments,
            "remaining_indices": remaining_indices,
            "removed_indices": removed_indices,
            "original_sufficiency": original_sufficiency,
            "pruned_sufficiency": current_sufficiency,
            "token_reduction": token_reduction,
            "segment_reduction": segment_reduction,
            "pruning_history": pruning_history
        }
        
        return pruned_explanation, pruning_details


class CounterfactualAnalyzer:
    """Class for counterfactual analysis of explanations"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def generate_counterfactual_mcqa(self, mcqa: MCQAInstance) -> Tuple[MCQAInstance, Dict[str, Any]]:
        """Generate a counterfactual version of the MCQA instance where the answer changes"""
        # Prompt to generate a counterfactual instance
        options_text = "\n".join([f"{key}. {value}" for key, value in mcqa.options.items()])
        
        prompt = f"""
        I'm going to show you a passage, question, and answer options. The correct answer is {mcqa.correct_option}.
        
        Passage: {mcqa.passage}
        
        Question: {mcqa.question}
        
        Options:
        {options_text}
        
        Task: Please modify the passage minimally so that a different option becomes the correct answer. 
        Keep the question and options exactly the same.
        
        1. Choose a different option (not {mcqa.correct_option}) to become the new correct answer
        2. Make minimal, precise changes to the passage to support this new answer
        3. Return your response in this format:
        
        New Correct Option: [letter]
        
        Modified Passage:
        [Your modified passage]
        
        Explanation of changes:
        [Brief explanation of what was changed and why it makes the new option correct]
        """
        
        response = self.llm_client.generate(prompt, temperature=0.7, max_tokens=1500)
        cf_text = response["text"].strip()
        
        # Extract the new correct option
        option_match = re.search(r"New Correct Option:\s*([A-D])", cf_text)
        new_option = option_match.group(1) if option_match else None
        
        # Extract the modified passage
        passage_match = re.search(r"Modified Passage:(.*?)(?:Explanation of changes:|$)", cf_text, re.DOTALL)
        modified_passage = passage_match.group(1).strip() if passage_match else None
        
        # Extract explanation of changes
        explanation_match = re.search(r"Explanation of changes:(.*?)$", cf_text, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else None
        
        # Create counterfactual instance
        if new_option and modified_passage:
            cf_mcqa = MCQAInstance(
                id=f"{mcqa.id}_cf" if mcqa.id else "cf",
                passage=modified_passage,
                question=mcqa.question,
                options=mcqa.options.copy(),
                correct_option=new_option
            )
            
            details = {
                "original_correct": mcqa.correct_option,
                "new_correct": new_option,
                "explanation_of_changes": explanation
            }
            
            return cf_mcqa, details
        else:
            # Fallback if extraction failed
            return mcqa, {"error": "Failed to generate valid counterfactual"}
    
    def analyze_explanation_differences(
        self, 
        original_explanation: Explanation, 
        cf_explanation: Explanation
    ) -> Dict[str, Any]:
        """Analyze the differences between original and counterfactual explanations"""
        # Prompt for analyzing the differences
        prompt = f"""
        I have two explanations for different multiple choice questions:
        
        Original Explanation (for option {original_explanation.chosen_option}):
        {original_explanation.text}
        
        Counterfactual Explanation (for option {cf_explanation.chosen_option}):
        {cf_explanation.text}
        
        Please analyze the key differences between these explanations:
        1. What evidence or reasoning elements appear uniquely in each explanation?
        2. What are the critical differences that explain why each supports a different answer choice?
        3. Are there any shared elements or reasoning patterns despite supporting different answers?
        
        Format your analysis as a structured comparison focusing on the most important differences.
        """
        
        response = self.llm_client.generate(prompt, temperature=0.3, max_tokens=1000)
        analysis = response["text"].strip()
        
        # Simple text-based similarity
        orig_tokens = set(original_explanation.text.lower().split())
        cf_tokens = set(cf_explanation.text.lower().split())
        
        jaccard_similarity = len(orig_tokens.intersection(cf_tokens)) / len(orig_tokens.union(cf_tokens))
        
        return {
            "textual_analysis": analysis,
            "jaccard_similarity": jaccard_similarity,
            "original_option": original_explanation.chosen_option,
            "cf_option": cf_explanation.chosen_option
        }