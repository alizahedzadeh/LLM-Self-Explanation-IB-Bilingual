from typing import List, Dict, Optional, Union, Any
import pandas as pd
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from model_utils import OpenRouterClient, GPT4OMiniClient
from data_structures import MCQAInstance, Explanation, ExperimentResult
from explanation_generation import IBGuidedExplanationGenerator, ExplanationRefinementSystem
from explanation_evaluation import ExplanationEvaluator
from explanation_bottleneck import ExplanationPruner, CounterfactualAnalyzer
from explanation_styles import ExplanationStyleGenerator, StyleMapper

class ExperimentRunner:
    """Main class for running experiments"""
    
    def __init__(self, predictor_api_key: str, judge_api_key: str, output_dir: str = "results"):
        # Initialize API clients
        self.predictor = OpenRouterClient(
            api_key=predictor_api_key, 
            model="anthropic/claude-3-haiku",  # Using Llama 4 equivalent
            rate_limit_delay=2  # Adjust based on API rate limits
        )
        
        self.judge = GPT4OMiniClient(
            api_key=judge_api_key,
            rate_limit_delay=1
        )
        
        # Initialize components
        self.generator = IBGuidedExplanationGenerator(self.predictor)
        self.evaluator = ExplanationEvaluator(self.judge)
        self.refiner = ExplanationRefinementSystem(self.predictor)
        self.pruner = ExplanationPruner(self.judge, self.evaluator)
        self.analyzer = CounterfactualAnalyzer(self.predictor)
        self.style_generator = ExplanationStyleGenerator(self.predictor)
        self.style_mapper = StyleMapper(self.evaluator)
        
        # Setup output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_mcqa_instances(self, file_path: str) -> List[MCQAInstance]:
        """Load MCQA instances from a JSON file"""
        with open(file_path, "r") as f:
            data = json.load(f)
        
        instances = []
        for item in data:
            instance = MCQAInstance.from_dict(item)
            instances.append(instance)
        
        return instances
    
    def save_results(self, results: List[ExperimentResult], experiment_name: str):
        """Save experiment results"""
        output_path = os.path.join(self.output_dir, f"{experiment_name}.json")
        with open(output_path, "w") as f:
            json.dump([result.to_dict() for result in results], f, indent=2)
    
    def run_idea1_experiment(self, mcqa_instances: List[MCQAInstance], n_samples: int = None):
        """Run experiment for Idea 1: IB-Guided Prompting"""
        if n_samples is not None:
            mcqa_instances = mcqa_instances[:n_samples]
        
        results = []
        
        for mcqa in tqdm(mcqa_instances, desc="Running IB-Guided Prompting"):
            # Generate explanations with different prompts
            standard_exp = self.generator.get_standard_explanation(mcqa)
            minimal_exp = self.generator.get_ib_minimal_explanation(mcqa)
            essential_exp = self.generator.get_ib_essential_explanation(mcqa)
            comparative_exp = self.generator.get_ib_comparative_explanation(mcqa)
            
            # Evaluate explanations
            self.evaluator.evaluate_explanation(mcqa, standard_exp)
            self.evaluator.evaluate_explanation(mcqa, minimal_exp)
            self.evaluator.evaluate_explanation(mcqa, essential_exp)
            self.evaluator.evaluate_explanation(mcqa, comparative_exp)
            
            # Create result
            result = ExperimentResult(
                mcqa_instance=mcqa,
                explanations=[standard_exp, minimal_exp, essential_exp, comparative_exp]
            )
            results.append(result)
        
        # Save results
        self.save_results(results, "idea1_ib_guided_prompting")
        
        # Visualize results
        self._visualize_idea1_results(results)
        
        return results
    
    def run_idea2_experiment(self, mcqa_instances: List[MCQAInstance], n_samples: int = None):
        """Run experiment for Idea 2: Iterative Self-Refinement"""
        if n_samples is not None:
            mcqa_instances = mcqa_instances[:n_samples]
        
        results = []
        
        for mcqa in tqdm(mcqa_instances, desc="Running Iterative Self-Refinement"):
            # Generate initial explanation
            initial_exp = self.generator.get_standard_explanation(mcqa)
            self.evaluator.evaluate_explanation(mcqa, initial_exp)
            
            # Perform iterative refinement
            refined_exps = self.refiner.perform_iterative_refinement(mcqa, initial_exp, max_iterations=2)
            
            # Evaluate all explanations
            for exp in refined_exps[1:]:  # Skip initial, already evaluated
                self.evaluator.evaluate_explanation(mcqa, exp)
            
            # Create result
            result = ExperimentResult(
                mcqa_instance=mcqa,
                explanations=[initial_exp] + refined_exps[1:]
            )
            results.append(result)
        
        # Save results
        self.save_results(results, "idea2_iterative_refinement")
        
        # Visualize results
        self._visualize_idea2_results(results)
        
        return results
    
    def run_idea3_experiment(self, mcqa_instances: List[MCQAInstance], n_samples: int = None):
        """Run experiment for Idea 3: Evaluation Framework"""
        if n_samples is not None:
            mcqa_instances = mcqa_instances[:n_samples]
        
        results = []
        
        for mcqa in tqdm(mcqa_instances, desc="Running Evaluation Framework"):
            # Generate explanations with different qualities
            good_exp = self.generator.get_ib_essential_explanation(mcqa)
            verbose_exp = self.generator.get_standard_explanation(mcqa)  # Typically more verbose
            
            # Create intentionally poor explanations
            prompt = f"""
            Question: {mcqa.question}
            
            Give a very vague and generic explanation for why option '{mcqa.correct_option}' might be correct,
            without using specific details from any passage. Keep it short but unhelpful.
            """
            response = self.predictor.generate(prompt, temperature=0.7, max_tokens=100)
            vague_text = response["text"].strip()
            
            vague_exp = Explanation(
                mcqa_id=mcqa.id if mcqa.id else "",
                chosen_option=mcqa.correct_option,
                text=vague_text,
                generation_method="intentionally_vague"
            )
            vague_exp.token_count = self.predictor.count_tokens(vague_text)
            
            # Evaluate all explanations
            self.evaluator.evaluate_explanation(mcqa, good_exp)
            self.evaluator.evaluate_explanation(mcqa, verbose_exp)
            self.evaluator.evaluate_explanation(mcqa, vague_exp)
            
            # Create result
            result = ExperimentResult(
                mcqa_instance=mcqa,
                explanations=[good_exp, verbose_exp, vague_exp]
            )
            results.append(result)
        
        # Save results
        self.save_results(results, "idea3_evaluation_framework")
        
        # Visualize results
        self._visualize_idea3_results(results)
        
        return results
    
    def run_idea4_experiment(self, mcqa_instances: List[MCQAInstance], n_samples: int = None):
        """Run experiment for Idea 4: Explanation Bottleneck"""
        if n_samples is not None:
            mcqa_instances = mcqa_instances[:n_samples]
        
        results = []
        
        for mcqa in tqdm(mcqa_instances, desc="Running Explanation Bottleneck"):
            # Generate standard explanation
            std_exp = self.generator.get_standard_explanation(mcqa)
            self.evaluator.evaluate_explanation(mcqa, std_exp)
            
            # Prune explanation
            pruned_exp, pruning_details = self.pruner.prune_explanation(mcqa, std_exp)
            self.evaluator.evaluate_explanation(mcqa, pruned_exp)
            
            # Generate counterfactual instance and explanation
            cf_mcqa, cf_details = self.analyzer.generate_counterfactual_mcqa(mcqa)
            cf_exp = self.generator.get_standard_explanation(cf_mcqa)
            self.evaluator.evaluate_explanation(cf_mcqa, cf_exp)
            
            # Analyze differences
            diff_analysis = self.analyzer.analyze_explanation_differences(std_exp, cf_exp)
            
            # Create result
            result = ExperimentResult(
                mcqa_instance=mcqa,
                explanations=[std_exp, pruned_exp, cf_exp],
                metadata={
                    "pruning_details": pruning_details,
                    "counterfactual_mcqa": cf_mcqa.to_dict(),
                    "counterfactual_details": cf_details,
                    "difference_analysis": diff_analysis
                }
            )
            results.append(result)
        
        # Save results
        self.save_results(results, "idea4_explanation_bottleneck")
        
        # Visualize results
        self._visualize_idea4_results(results)
        
        return results
    
    def run_idea5_experiment(self, mcqa_instances: List[MCQAInstance], n_samples: int = None):
        """Run experiment for Idea 5: Style Mapping"""
        if n_samples is not None:
            mcqa_instances = mcqa_instances[:n_samples]
        
        results = []
        style_mappings = []
        
        for mcqa in tqdm(mcqa_instances, desc="Running Style Mapping"):
            # Generate explanations in different styles
            style_explanations = self.style_generator.generate_multiple_styles(mcqa)
            
            # Evaluate and map explanations
            style_mapping = self.style_mapper.map_multiple_explanations(mcqa, style_explanations)
            style_mappings.append(style_mapping)
            
            # Create result
            result = ExperimentResult(
                mcqa_instance=mcqa,
                explanations=list(style_explanations.values()),
                metadata={"style_mapping": style_mapping}
            )
            results.append(result)
        
        # Visualize styles on S-C plane
        self.style_mapper.visualize_sc_plane(style_mappings)
        
        # Save results
        self.save_results(results, "idea5_style_mapping")
        
        return results
    
    def _visualize_idea1_results(self, results: List[ExperimentResult]):
        """Visualize results for Idea 1"""
        # Extract data
        data = []
        for result in results:
            for exp in result.explanations:
                data.append({
                    "Method": exp.generation_method,
                    "Sufficiency": exp.sufficiency_score,
                    "Conciseness": exp.conciseness_score,
                    "Tokens": exp.token_count
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Calculate averages
        avg_df = df.groupby("Method").agg({
            "Sufficiency": "mean",
            "Conciseness": "mean",
            "Tokens": "mean"
        }).reset_index()
        
        # Plot sufficiency and conciseness
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Method", y="value", hue="metric", data=pd.melt(
            avg_df, 
            id_vars=["Method"], 
            value_vars=["Sufficiency", "Conciseness"],
            var_name="metric"
        ))
        plt.title("Sufficiency and Conciseness by Explanation Method")
        plt.ylabel("Score")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "idea1_sc_scores.png"))
        plt.close()
        
        # Plot token counts
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Method", y="Tokens", data=avg_df)
        plt.title("Average Token Count by Explanation Method")
        plt.ylabel("Tokens")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "idea1_token_counts.png"))
        plt.close()
    
    def _visualize_idea2_results(self, results: List[ExperimentResult]):
        """Visualize results for Idea 2"""
        # Extract data
        data = []
        for result in results:
            for i, exp in enumerate(result.explanations):
                iteration = 0
                if "_suff_refined" in exp.generation_method:
                    iteration = (i + 1) // 2
                    subtype = "Sufficiency"
                elif "_conc_refined" in exp.generation_method:
                    iteration = i // 2
                    subtype = "Conciseness"
                else:
                    subtype = "Initial"
                
                data.append({
                    "MCQA_ID": exp.mcqa_id,
                    "Iteration": iteration,
                    "Type": subtype,
                    "Sufficiency": exp.sufficiency_score,
                    "Conciseness": exp.conciseness_score,
                    "Tokens": exp.token_count
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Plot sufficiency over iterations
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="Iteration", y="Sufficiency", hue="Type", marker="o")
        plt.title("Sufficiency Score Over Refinement Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Sufficiency Score")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "idea2_sufficiency.png"))
        plt.close()
        
        # Plot conciseness over iterations
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="Iteration", y="Conciseness", hue="Type", marker="o")
        plt.title("Conciseness Score Over Refinement Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Conciseness Score")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "idea2_conciseness.png"))
        plt.close()
        
        # Plot token counts over iterations
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="Iteration", y="Tokens", hue="Type", marker="o")
        plt.title("Token Count Over Refinement Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Tokens")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "idea2_tokens.png"))
        plt.close()
    
    def _visualize_idea3_results(self, results: List[ExperimentResult]):
        """Visualize results for Idea 3"""
        # Extract data
        data = []
        for result in results:
            for exp in result.explanations:
                data.append({
                    "Method": exp.generation_method,
                    "Sufficiency": exp.sufficiency_score,
                    "Conciseness": exp.conciseness_score,
                    "Tokens": exp.token_count
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Create sufficiency-conciseness plot
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            data=df, 
            x="Sufficiency", 
            y="Conciseness", 
            hue="Method",
            size="Tokens",
            sizes=(50, 200),
            alpha=0.7
        )
        plt.title("Explanation Quality on the Sufficiency-Conciseness Plane")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "idea3_sc_plane.png"))
        plt.close()
    
    def _visualize_idea4_results(self, results: List[ExperimentResult]):
        """Visualize results for Idea 4"""
        # Extract pruning data
        pruning_data = []
        for result in results:
            if "pruning_details" not in result.metadata:
                continue
                
            details = result.metadata["pruning_details"]
            if not details.get("prunable", False):
                continue
            
            original_exp = next(e for e in result.explanations if e.generation_method == "standard")
            pruned_exp = next((e for e in result.explanations if "bottleneck" in e.generation_method), None)
            
            if not pruned_exp:
                continue
            
            pruning_data.append({
                "MCQA_ID": result.mcqa_instance.