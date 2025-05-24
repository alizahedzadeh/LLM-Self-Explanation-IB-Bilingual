from typing import List, Dict, Optional, Union, Any
from model_utils import LLMClient
from data_structures import MCQAInstance, Explanation
import re

class ExplanationGenerator:
    """Base class for generating explanations"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def generate_explanation(self, mcqa: MCQAInstance, prompt_template: str) -> Explanation:
        """Generate an explanation using a specific prompt template"""
        # Format the prompt template with MCQA instance details
        options_text = "\n".join([f"{key}. {value}" for key, value in mcqa.options.items()])
        prompt = prompt_template.format(
            passage=mcqa.passage,
            question=mcqa.question,
            options=options_text,
            correct_option=mcqa.correct_option,
            correct_option_text=mcqa.options[mcqa.correct_option]
        )
        
        # Generate completion
        response = self.llm_client.generate(prompt, temperature=0.3, max_tokens=1000)
        explanation_text = response["text"].strip()
        
        # Create and return explanation
        explanation = Explanation(
            mcqa_id=mcqa.id if mcqa.id else "",
            chosen_option=mcqa.correct_option,
            text=explanation_text,
            generation_method=self._get_generation_method(prompt_template)
        )
        explanation.token_count = self.llm_client.count_tokens(explanation_text)
        
        return explanation
    
    def _get_generation_method(self, prompt_template: str) -> str:
        """Extract a short name for the generation method from the prompt template"""
        # This is a simple implementation; you might want to use a more robust method
        if "minimal reasoning" in prompt_template.lower():
            return "minimal"
        elif "essential information" in prompt_template.lower():
            return "essential"
        elif "briefly explain" in prompt_template.lower():
            return "brief"
        elif "step-by-step" in prompt_template.lower():
            return "step_by_step"
        elif "evidence sentences" in prompt_template.lower():
            return "evidence_extract"
        elif "each other option" in prompt_template.lower():
            return "full_contrastive"
        elif "most likely incorrect option" in prompt_template.lower():
            return "partial_contrastive"
        elif "concise summary" in prompt_template.lower():
            return "concise"
        else:
            return "standard"


class IBGuidedExplanationGenerator(ExplanationGenerator):
    """Generator for IB-guided explanations"""
    
    def get_standard_explanation(self, mcqa: MCQAInstance) -> Explanation:
        """Generate a standard explanation"""
        prompt = """
        Passage: {passage}
        
        Question: {question}
        
        Options:
        {options}
        
        Explain why option '{correct_option}' ({correct_option_text}) is the correct answer.
        """
        return self.generate_explanation(mcqa, prompt)
    
    def get_ib_minimal_explanation(self, mcqa: MCQAInstance) -> Explanation:
        """Generate an IB-guided minimal explanation"""
        prompt = """
        Passage: {passage}
        
        Question: {question}
        
        Options:
        {options}
        
        Provide the minimal reasoning, citing evidence from the passage, that leads to selecting option '{correct_option}' ({correct_option_text}) for the question. Focus solely on justifying this choice.
        """
        return self.generate_explanation(mcqa, prompt)
    
    def get_ib_essential_explanation(self, mcqa: MCQAInstance) -> Explanation:
        """Generate an IB-guided explanation focusing on essential information"""
        prompt = """
        Passage: {passage}
        
        Question: {question}
        
        Options:
        {options}
        
        Based only on the essential information in the passage needed to answer the question, explain concisely why option '{correct_option}' ({correct_option_text}) is the correct answer. Avoid passage details not strictly necessary for this justification.
        """
        return self.generate_explanation(mcqa, prompt)
    
    def get_ib_comparative_explanation(self, mcqa: MCQAInstance) -> Explanation:
        """Generate an IB-guided explanation with minimal comparison"""
        prompt = """
        Passage: {passage}
        
        Question: {question}
        
        Options:
        {options}
        
        Briefly explain why option '{correct_option}' ({correct_option_text}) is correct, contrasting it only with the most plausible incorrect option if necessary for clarity.
        """
        return self.generate_explanation(mcqa, prompt)


class ExplanationRefinementSystem:
    """System for iterative self-refinement of explanations"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def refine_for_sufficiency(
        self, 
        mcqa: MCQAInstance, 
        current_explanation: Explanation
    ) -> Explanation:
        """Refine an explanation to improve sufficiency"""
        options_text = "\n".join([f"{key}. {value}" for key, value in mcqa.options.items()])
        
        prompt = f"""
        Review the following explanation for why option '{mcqa.correct_option}' is the answer to the given question.
        
        Question: {mcqa.question}
        
        Options:
        {options_text}
        
        Current explanation:
        "{current_explanation.text}"
        
        Based only on this explanation, is it clear and convincing that '{mcqa.correct_option}' is the correct choice? 
        If not, what critical reasoning or passage evidence is missing?
        
        Please provide an improved explanation that better justifies selecting '{mcqa.correct_option}' ({mcqa.options[mcqa.correct_option]}).
        """
        
        response = self.llm_client.generate(prompt, temperature=0.3, max_tokens=1200)
        improved_explanation = response["text"].strip()
        
        # Extract just the explanation part, skipping meta-commentary
        # This is a simple extractor that might need refinement
        match = re.search(r"(?:Improved explanation:|New explanation:)(.*)", improved_explanation, re.DOTALL)
        if match:
            improved_explanation = match.group(1).strip()
        
        new_explanation = Explanation(
            mcqa_id=mcqa.id if mcqa.id else "",
            chosen_option=mcqa.correct_option,
            text=improved_explanation,
            generation_method=f"{current_explanation.generation_method}_suff_refined"
        )
        new_explanation.token_count = self.llm_client.count_tokens(improved_explanation)
        
        return new_explanation
    
    def refine_for_conciseness(
        self, 
        mcqa: MCQAInstance, 
        current_explanation: Explanation
    ) -> Explanation:
        """Refine an explanation to improve conciseness"""
        prompt = f"""
        Review the following explanation for why option '{mcqa.correct_option}' is the answer to the question "{mcqa.question}":
        
        "{current_explanation.text}"
        
        Does this explanation contain any information or reasoning steps from the passage that are not strictly necessary to justify selecting option '{mcqa.correct_option}'? 
        
        Please provide a revised, more concise explanation that still clearly supports '{mcqa.correct_option}' as the correct answer.
        """
        
        response = self.llm_client.generate(prompt, temperature=0.3, max_tokens=1000)
        concise_explanation = response["text"].strip()
        
        # Extract just the explanation part, skipping meta-commentary
        match = re.search(r"(?:Revised explanation:|Concise explanation:|Here's a more concise explanation:)(.*)", concise_explanation, re.DOTALL)
        if match:
            concise_explanation = match.group(1).strip()
        
        new_explanation = Explanation(
            mcqa_id=mcqa.id if mcqa.id else "",
            chosen_option=mcqa.correct_option,
            text=concise_explanation,
            generation_method=f"{current_explanation.generation_method}_conc_refined"
        )
        new_explanation.token_count = self.llm_client.count_tokens(concise_explanation)
        
        return new_explanation
    
    def perform_iterative_refinement(
        self, 
        mcqa: MCQAInstance, 
        initial_explanation: Explanation, 
        max_iterations: int = 3
    ) -> List[Explanation]:
        """Perform iterative refinement, alternating between sufficiency and conciseness"""
        explanations = [initial_explanation]
        current = initial_explanation
        
        for i in range(1, max_iterations + 1):
            # First improve sufficiency
            suff_improved = self.refine_for_sufficiency(mcqa, current)
            explanations.append(suff_improved)
            
            # Then improve conciseness
            conc_improved = self.refine_for_conciseness(mcqa, suff_improved)
            explanations.append(conc_improved)
            
            # Update current
            current = conc_improved
        
        return explanations