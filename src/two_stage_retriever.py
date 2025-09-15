import os
from typing import List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

from retriever import BM25


def split_paragraphs(text: str, max_chars: int = 400) -> List[str]:
    parts = []
    buf = []
    cur = 0
    for seg in text.split("\n"):
        seg = seg.strip()
        if not seg:
            continue
        if cur + len(seg) + (1 if buf else 0) > max_chars and buf:
            parts.append(" ".join(buf))
            buf = [seg]
            cur = len(seg)
        else:
            buf.append(seg)
            cur += len(seg) + (1 if buf else 0)
    if buf:
        parts.append(" ".join(buf))
    if not parts:
        return [text[:max_chars]]
    return parts


class TwoStageRetriever:
    def __init__(self, args):
        self.bm25 = BM25("wiki" if "es_index_name" not in args else args.es_index_name)
        self.initial_bm25 = getattr(args, "initial_bm25", 100)
        self.initial_dense = getattr(args, "initial_dense", 100)
        self.paragraph_max_chars = getattr(args, "paragraph_max_chars", 400)
        self.max_paragraphs_to_score = getattr(args, "max_paragraphs_to_score", 200)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.top_k_fallback = getattr(args, "top_k_paragraphs", getattr(args, "retrieve_topk", 3))

        self.dense_model_name = getattr(args, "dense_model_name", "intfloat/e5-base-v2")
        self.cross_encoder_name = getattr(args, "cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.dense_model = SentenceTransformer(self.dense_model_name, device=self.device)
        self.cross_encoder = CrossEncoder(self.cross_encoder_name, device=self.device)

        self.index_dir = getattr(args, "dense_index_dir", None)
        self.faiss_index = None
        self.texts = None
        if self.index_dir:
            index_path = os.path.join(self.index_dir, "index.faiss")
            texts_path = os.path.join(self.index_dir, "texts.txt")
            if os.path.exists(index_path) and os.path.exists(texts_path):
                self.faiss_index = faiss.read_index(index_path)
                with open(texts_path, "r") as f:
                    self.texts = [line.rstrip("\n") for line in f]

    def dense_search(self, qry: str, topk: int) -> List[str]:
        if self.faiss_index is None or self.texts is None:
            return []
        q = qry
        if "e5" in self.dense_model_name.lower():
            q = f"query: {qry}"
        q_emb = self.dense_model.encode([q], normalize_embeddings=True)
        D, I = self.faiss_index.search(np.asarray(q_emb, dtype="float32"), topk)
        idxs = I[0].tolist()
        return [self.texts[i] for i in idxs if i >= 0]

    def __call__(self, qry: str, topk: int = 3) -> List[str]:
        bm25_docs = self.bm25.lexical_search(qry, top_hits=self.initial_bm25)
        dense_docs = self.dense_search(qry, self.initial_dense)

        seen = set()
        candidates = []
        for d in bm25_docs + dense_docs:
            if d not in seen:
                seen.add(d)
                candidates.append(d)
            if len(candidates) >= self.initial_bm25 + self.initial_dense:
                break

        paras = []
        for d in candidates:
            for p in split_paragraphs(d, self.paragraph_max_chars):
                paras.append(p)
                if len(paras) >= self.max_paragraphs_to_score:
                    break
            if len(paras) >= self.max_paragraphs_to_score:
                break

        if not paras:
            return bm25_docs[:max(1, topk)]

        pairs = [(qry, p) for p in paras]
        scores = self.cross_encoder.predict(pairs)
        order = np.argsort(-np.asarray(scores))
        k = min(topk if topk is not None else self.top_k_fallback, len(paras))
        return [paras[i] for i in order[:k]]




