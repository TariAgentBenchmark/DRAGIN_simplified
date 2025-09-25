from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple
import statistics

import torch


def _unique_token_ids(tokenizer, variants: Sequence[str]) -> List[int]:
    ids: List[int] = []
    seen = set()
    for variant in variants:
        tokens = tokenizer.encode(variant, add_special_tokens=False)
        if tokens:
            token_id = tokens[0]
            if token_id not in seen:
                seen.add(token_id)
                ids.append(token_id)
    return ids


def _single_token_log_probs(model, tokenizer, prompt: str) -> torch.Tensor:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    return torch.log_softmax(logits, dim=-1)


def _agent1_answer(generator, question: str, document: str, max_len: int) -> str:
    prompt = (
        "Use the following document to answer the question.\n"
        f"Document:\n{document}\n\n"
        f"Question: {question}\nAnswer:"
    )
    ended, generated = generator.simply_generate(prompt, max_length=max_len)
    return generated.strip()


def _agent2_log_odds(generator, question: str, document: str, answer: str) -> float:
    prompt = (
        "You are a noisy document evaluator. Given a Document, a Question, and an LLM Answer, "
        "decide whether (1) the Document supplies the information to answer the Question and "
        "(2) the LLM Answer is directly grounded in the Document. Respond with 'Yes' or 'No' only.\n"
        f"Document:\n{document}\n\n"
        f"Question: {question}\n"
        f"LLM Answer: {answer}\n"
        "Judgment (Yes/No):"
    )
    log_probs = _single_token_log_probs(generator.model, generator.tokenizer, prompt)
    yes_ids = _unique_token_ids(generator.tokenizer, [" Yes", "Yes"])
    no_ids = _unique_token_ids(generator.tokenizer, [" No", "No"])
    yes_lp = max((log_probs[idx].item() for idx in yes_ids), default=float("-inf"))
    no_lp = max((log_probs[idx].item() for idx in no_ids), default=float("-inf"))
    return yes_lp - no_lp


def main_rag_filter(
    generator,
    question: str,
    documents: Sequence[str],
    max_predict_len: int = 48,
    sigma_scale: float = 0.0,
) -> Tuple[List[str], List[float], float]:
    if not documents:
        return [], [], 0.0

    answers = [_agent1_answer(generator, question, doc, max_predict_len) for doc in documents]
    scores = [_agent2_log_odds(generator, question, doc, ans) for doc, ans in zip(documents, answers)]

    mean_score = statistics.mean(scores)
    std_score = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    threshold = mean_score - sigma_scale * std_score

    kept = [(doc, score) for doc, score in zip(documents, scores) if score >= threshold]
    kept.sort(key=lambda item: item[1], reverse=True)

    ordered_docs = [doc for doc, _ in kept]
    return ordered_docs, scores, threshold



