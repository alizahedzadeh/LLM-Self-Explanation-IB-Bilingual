from typing import List, Dict, Optional, Union, Any
from model_utils import LLMClient
from data_structures import MCQAInstance, Explanation
from explanation_evaluation import ExplanationEvaluator

class ExplanationStyleGenerator:
    """Generator for different explanation styles"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def generate_explanation(self, mcqa: MCQAInstance, style: str) -> Explanation:
        """Generate an explanation using a specific style"""
        options_text = "\n".join([f"{key}. {value}" for key, value in mcqa.options.items()])
        
        # Select prompt template based on style
        if style == "step_by_step":
            prompt_template = """
            Passage: {passage}
            
            Question: {question}
            
            Options:
            {options}
            
            Explain step-by-step why option '{correct_option}' ({correct_option_text}) is correct.
            """
        
        elif style == "evidence_extract":
            prompt_template = """
            Passage: {passage}
            
            Question: {question}
            
            Options:
            {options}
            
            Extract only the key evidence sentences from the passage that support option '{correct_option}' ({correct_option_text}).
            """
        
        elif style == "full_contrastive":
            prompt_template = """
            Passage: {passage}
            
            Question: {question}
            
            Options:
            {options}
            
            Explain why option '{correct_option}' ({correct_option_text}) is correct and briefly explain why each other option is incorrect.
            """
        
        elif style == "partial_contrastive":
            prompt_template = """
            Passage: {passage}
            
            Question: {question}
            
            Options:
            {options}
            
            Explain why option '{correct_option}' ({correct_option_text}) is correct, contrasting it only with the most likely incorrect option.
            """
        
        elif style == "concise":
            prompt_template = """
            Passage: {passage}
            
            Question: {question}
            
            Options:
            {options}
            
            Provide a concise summary justifying the choice of option '{correct_option}' ({correct_option_text}).
            """
        
        else:  # default style
            prompt_template = """
            Passage: {passage}
            
            Question: {question}
            
            Options:
            {options}
            
            Explain why option '{correct_option}' ({correct_option_text}) is the correct answer.
            """
            style = "default"
        
        # Format the prompt
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
            generation_method=style
        )
        explanation.token_count = self.llm_client.count_tokens(explanation_text)
        
        return explanation
    
    def generate_multiple_styles(self, mcqa: MCQAInstance) -> Dict[str, Explanation]:
        """Generate explanations in multiple styles"""
        styles = [
            "default",
            "step_by_step",
            "evidence_extract",
            "full_contrastive",
            "partial_contrastive",
            "concise"
        ]
        
        result = {}
        for style in styles:
            result[style] = self.generate_explanation(mcqa, style)
        
        return result


class StyleMapper:
    """Maps explanation styles to the S-C plane"""
    
    def __init__(self, evaluator: ExplanationEvaluator):
        self.evaluator = evaluator
    
    def map_explanation_to_sc(
        self, 
        mcqa: MCQAInstance, 
        explanation: Explanation
    ) -> Dict[str, float]:
        """Map an explanation to the S-C plane"""
        # Evaluate sufficiency
        sufficiency, _ = self.evaluator.evaluate_sufficiency(mcqa, explanation)
        explanation.sufficiency_score = sufficiency
        
        # Evaluate conciseness
        conciseness, _ = self.evaluator.evaluate_conciseness(explanation)
        explanation.conciseness_score = conciseness
        
        return {
            "style": explanation.generation_method,
            "sufficiency": sufficiency,
            "conciseness": conciseness,
            "token_count": explanation.token_count
        }
    
    def map_multiple_explanations(
        self, 
        mcqa: MCQAInstance, 
        explanations: Dict[str, Explanation]
    ) -> Dict[str, Dict[str, float]]:
        """Map multiple explanations to the S-C plane"""
        result = {}
        for style, explanation in explanations.items():
            result[style] = self.map_explanation_to_sc(mcqa, explanation)
        
        return result
    
    def visualize_sc_plane(self, style_mappings: List[Dict[str, Dict[str, float]]]):
        """Visualize explanation styles on the S-C plane"""
        # Aggregate data across MCQAs
        styles = {}
        
        for mapping in style_mappings:
            for style, data in mapping.items():
                if style not in styles:
                    styles[style] = {
                        "sufficiency": [],
                        "conciseness": [],
                        "token_count": []
                    }
                
                styles[style]["sufficiency"].append(data["sufficiency"])
                styles[style]["conciseness"].append(data["conciseness"])
                styles[style]["token_count"].append(data["token_count"])
        
        # Calculate averages
        for style in styles:
            styles[style]["avg_sufficiency"] = np.mean(styles[style]["sufficiency"])
            styles[style]["avg_conciseness"] = np.mean(styles[style]["conciseness"])
            styles[style]["avg_token_count"] = np.mean(styles[style]["token_count"])
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        # Scatter plot of styles
        for style, data in styles.items():
            plt.scatter(
                data["avg_sufficiency"], 
                data["avg_conciseness"], 
                s=100, 
                label=style,
                alpha=0.7
            )
        
        # Add style labels
        for style, data in styles.items():
            plt.annotate(
                style,
                (data["avg_sufficiency"], data["avg_conciseness"]),
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("Sufficiency Score")
        plt.ylabel("Conciseness Score")
        plt.title("Explanation Styles on the Sufficiency-Conciseness Plane")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save figure
        plt.savefig("explanation_style_map.png", dpi=300, bbox_inches='tight')
        plt.close()