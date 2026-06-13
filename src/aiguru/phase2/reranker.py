"""Official Transformers inference adapter for BGE reranker models."""

from __future__ import annotations

from typing import List, Sequence, Tuple


class BGEReranker:
    def __init__(self, model_name: str, device: str = "cuda", max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval().to(device)
        if device.startswith("cuda"):
            self.model.half()

    def predict(
        self,
        pairs: Sequence[Tuple[str, str]],
        batch_size: int = 16,
        show_progress_bar: bool = False,
    ) -> List[float]:
        del show_progress_bar
        scores: List[float] = []
        for start in range(0, len(pairs), batch_size):
            batch = list(pairs[start : start + batch_size])
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            ).to(self.device)
            with self.torch.inference_mode():
                logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
            scores.extend(float(value) for value in logits.cpu().tolist())
        return scores
