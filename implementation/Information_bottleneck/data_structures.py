from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Any

@dataclass
class MCQAInstance:
    """Represents a Multiple Choice QA instance"""
    passage: str
    question: str
    options: Dict[str, str]  # e.g., {"A": "option text", "B": "option text"...}
    correct_option: str  # e.g., "A", "B", etc.
    id: str = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "passage": self.passage,
            "question": self.question,
            "options": self.options,
            "correct_option": self.correct_option
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCQAInstance':
        return cls(
            id=data.get("id"),
            passage=data["passage"],
            question=data["question"],
            options=data["options"],
            correct_option=data["correct_option"]
        )


@dataclass
class Explanation:
    """Represents an explanation for an MCQA answer"""
    text: str
    mcqa_id: str
    chosen_option: str
    generation_method: str = "default"
    
    # Evaluation metrics (to be filled in later)
    sufficiency_score: Optional[float] = None
    conciseness_score: Optional[float] = None
    token_count: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mcqa_id": self.mcqa_id,
            "chosen_option": self.chosen_option,
            "text": self.text,
            "generation_method": self.generation_method,
            "sufficiency_score": self.sufficiency_score,
            "conciseness_score": self.conciseness_score,
            "token_count": self.token_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Explanation':
        return cls(
            mcqa_id=data["mcqa_id"],
            chosen_option=data["chosen_option"],
            text=data["text"],
            generation_method=data.get("generation_method", "default"),
            sufficiency_score=data.get("sufficiency_score"),
            conciseness_score=data.get("conciseness_score"),
            token_count=data.get("token_count")
        )


@dataclass
class ExperimentResult:
    """Represents results of an experiment"""
    mcqa_instance: MCQAInstance
    explanations: List[Explanation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mcqa_instance": self.mcqa_instance.to_dict(),
            "explanations": [exp.to_dict() for exp in self.explanations],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentResult':
        result = cls(
            mcqa_instance=MCQAInstance.from_dict(data["mcqa_instance"]),
            metadata=data.get("metadata", {})
        )
        result.explanations = [Explanation.from_dict(exp) for exp in data["explanations"]]
        return result