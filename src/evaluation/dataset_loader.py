"""
Dataset Loader Module.

Loads, validates, and serializes the golden benchmark dataset containing
annotated questions, reference answers, and ground-truth chunk IDs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass, asdict


@dataclass
class BenchmarkQuestion:
    """
    Evaluation instance conforming to Section 5.1 of specification:
    {
      "question_id": "Q001",
      "question": "What is the main objective of the system?",
      "reference_answer": "Retrieve relevant information and answer questions.",
      "relevant_chunk_ids": ["doc01_chunk_002", "doc01_chunk_003"]
    }
    """
    question_id: str
    question: str
    reference_answer: str
    relevant_chunk_ids: List[str]
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert BenchmarkQuestion to dictionary representation."""
        data = {
            "question_id": self.question_id,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "relevant_chunk_ids": self.relevant_chunk_ids,
        }
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BenchmarkQuestion:
        """Construct BenchmarkQuestion from raw dictionary."""
        q_id = str(data.get("question_id", "")).strip()
        q_text = str(data.get("question", "")).strip()
        ref_ans = str(data.get("reference_answer", "")).strip()
        raw_ids = data.get("relevant_chunk_ids", [])
        if not isinstance(raw_ids, list):
            raw_ids = [str(raw_ids)]
        clean_ids = [str(cid).strip() for cid in raw_ids if str(cid).strip()]

        if not q_id:
            raise ValueError("Benchmark question item is missing 'question_id'")
        if not q_text:
            raise ValueError(f"Question {q_id} has empty 'question' text")

        return cls(
            question_id=q_id,
            question=q_text,
            reference_answer=ref_ans,
            relevant_chunk_ids=clean_ids,
            metadata=data.get("metadata")
        )


class DatasetLoader:
    """
    Loads and validates benchmark datasets from disk or memory.
    """

    @staticmethod
    def load_from_file(file_path: Union[str, Path]) -> List[BenchmarkQuestion]:
        """
        Load evaluation dataset from a JSON file.

        Args:
            file_path: Path to the JSON benchmark file.

        Returns:
            List of validated BenchmarkQuestion objects.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark dataset not found: {path}")

        try:
            content = path.read_text(encoding="utf-8")
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON in benchmark dataset '{file_path}': {e}") from e

        if not isinstance(data, list):
            raise ValueError(f"Benchmark dataset JSON must be a list of objects, got {type(data).__name__}")

        questions: List[BenchmarkQuestion] = []
        seen_ids = set()

        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Dataset item at index {idx} must be a dictionary")
            question_obj = BenchmarkQuestion.from_dict(item)
            if question_obj.question_id in seen_ids:
                raise ValueError(f"Duplicate question_id detected in dataset: '{question_obj.question_id}'")
            seen_ids.add(question_obj.question_id)
            questions.append(question_obj)

        return questions

    @staticmethod
    def save_to_file(questions: List[BenchmarkQuestion], file_path: Union[str, Path]) -> None:
        """
        Serialize benchmark questions list to a JSON file.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw_list = [q.to_dict() for q in questions]
        path.write_text(json.dumps(raw_list, indent=2, ensure_ascii=False), encoding="utf-8")
