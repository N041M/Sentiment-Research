from __future__ import annotations

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sentiment_signal.config import settings

FINBERT = "ProsusAI/finbert"
FOMC_ROBERTA = "gtfintechlab/FOMC-RoBERTa"
DISTILBERT_SST = "distilbert-base-uncased-finetuned-sst-2-english"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# Stored in statement_analysis.model_version. Bump when the scoring method changes
# so step3 knows which rows are stale and must be re-scored.
FINBERT_CHUNKED_VERSION = "ProsusAI/finbert+chunk-meanpool-v1"

# Transformer context limit; leave room for the [CLS]/[SEP] special tokens.
_MAX_TOKENS = 512
_CHUNK_TOKENS = _MAX_TOKENS - 2

# FOMC-RoBERTa label mapping. The published model ships a generic
# id2label={0:LABEL_0, 1:LABEL_1, 2:LABEL_2}: the semantics live ONLY in the model
# card, which documents label_0=dovish, label_1=hawkish, label_2=neutral
# (gtfintechlab/FOMC-RoBERTa; Shah, Paturi & Chava, ACL 2023; verified against the
# published config.json, June 2026). Getting this order wrong silently INVERTS the
# entire stance signal, so resolve indices via resolve_hawk_dove_indices rather than
# trusting a positional assumption. hawkish_score = P(hawkish) - P(dovish).
_FOMC_DOVE_IDX = 0
_FOMC_HAWK_IDX = 1
_FOMC_NEUTRAL_IDX = 2


def resolve_hawk_dove_indices(id2label: dict) -> tuple[int, int]:
    """Return (hawk_idx, dove_idx) for a hawkish/dovish classifier's output.

    Prefers descriptive class names in the model config (`hawkish`/`dovish`); falls
    back to the documented gtfintechlab/FOMC-RoBERTa order (label_0=dovish,
    label_1=hawkish) when the config carries only generic LABEL_N names — as the
    published one does. This keeps the sign correct regardless of how the loaded
    model happens to name its classes.
    """
    lower = {int(k): str(v).lower() for k, v in id2label.items()}
    hawk = next((k for k, v in lower.items() if "hawk" in v), None)
    dove = next((k for k, v in lower.items() if "dov" in v), None)
    if hawk is not None and dove is not None:
        return hawk, dove
    return _FOMC_HAWK_IDX, _FOMC_DOVE_IDX


def _best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def chunk_windows(ids: list[int], window: int) -> list[list[int]]:
    """Split a token-id list into consecutive windows of at most `window` ids."""
    if not ids or window <= 0:
        return []
    return [ids[i : i + window] for i in range(0, len(ids), window)]


def weighted_average(vectors: list, weights: list) -> np.ndarray:
    """Weighted mean of row vectors. Falls back to equal weights if all are zero."""
    a = np.asarray(vectors, dtype=float)
    w = np.asarray(weights, dtype=float)
    if w.sum() == 0:
        w = np.ones_like(w)
    return np.average(a, axis=0, weights=w)


class NLPPipeline:
    """Loads a single transformer and runs batch sentiment scoring + embedding."""

    def __init__(self, model_name: str = FINBERT, device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device or _best_device()
        # float16 matmul is not implemented on CPU (raises "addmm_impl_cpu_ not
        # implemented for 'Half'"), so only use half precision on a GPU device.
        self.use_fp16 = self.device in ("cuda", "mps")
        logger.info(f"NLPPipeline: model={model_name}, device={self.device}, fp16={self.use_fp16}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # We chunk manually and always pass an explicit max_length when truncating, so
        # disable the tokenizer's default length check — otherwise encoding a long
        # document in _chunk_text logs a "sequence too long" warning for every document.
        self.tokenizer.model_max_length = int(1e9)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if self.use_fp16:
            model = model.half()  # halves VRAM, negligible accuracy loss at inference
        self.model = model.to(self.device)
        self.model.eval()

    def _probs_to_score_label(self, probs) -> tuple[float, str]:
        """Map a probability vector to a polarity score (pos − neg) and a label."""
        probs = np.asarray(probs, dtype=float)
        id2label = self.model.config.id2label
        label = id2label[int(probs.argmax())].lower()
        pos_idx = next((k for k, v in id2label.items() if "pos" in v.lower()), None)
        neg_idx = next((k for k, v in id2label.items() if "neg" in v.lower()), None)
        score = (
            float(probs[pos_idx] - probs[neg_idx])
            if pos_idx is not None and neg_idx is not None
            else 0.0
        )
        return score, label

    def score_batch(self, texts: list[str]) -> list[dict]:
        """Return sentiment score ∈ [-1, 1], label, and raw probs for each text.

        Each text is truncated to one 512-token window — use analyze_documents for
        long documents that must be scored in full.
        """
        results: list[dict] = []
        for i in range(0, len(texts), settings.nlp_batch_size):
            batch = texts[i : i + settings.nlp_batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", truncation=True, padding=True, max_length=_MAX_TOKENS
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = logits.float().softmax(dim=-1).cpu()
            for prob_row in probs:
                score, label = self._probs_to_score_label(prob_row)
                results.append(
                    {"sentiment_score": score, "sentiment_label": label, "probs": prob_row.tolist()}
                )
        return results

    def _chunk_text(self, text: str, tokenizer=None) -> list[tuple[str, int]]:
        """Split text into (chunk_text, token_count) windows of ≤ _CHUNK_TOKENS tokens.

        Chunks with `tokenizer` (defaults to the main model's). The FOMC model has a
        different tokenizer, so its chunk boundaries must be computed with its own.
        """
        tok = tokenizer or self.tokenizer
        ids = tok.encode(text or "", add_special_tokens=False)
        windows = chunk_windows(ids, _CHUNK_TOKENS)
        return [(tok.decode(w, skip_special_tokens=True), len(w)) for w in windows]

    def analyze_documents(self, texts: list[str]) -> list[dict]:
        """Score + embed full documents by chunking, then token-weighted aggregation.

        FinBERT truncates at 512 tokens; most speeches/orders are far longer, so a
        single pass scores only the opening. This splits each document into 512-token
        chunks, scores and embeds every chunk, then aggregates (token-weighted mean of
        the class probabilities and of the CLS embeddings) into one document result.

        Returns per-document dicts: sentiment_score, sentiment_label, probs (aggregated),
        embedding (768-d, mean-pooled), n_chunks.
        """
        # Flatten all chunks across documents into one list, tracking per-doc spans.
        all_chunks: list[str] = []
        weights: list[int] = []
        spans: list[tuple[int, int]] = []
        for text in texts:
            chunks = self._chunk_text(text) or [("", 1)]  # placeholder for empty text
            start = len(all_chunks)
            for chunk_text, n_tokens in chunks:
                all_chunks.append(chunk_text)
                weights.append(n_tokens)
            spans.append((start, len(all_chunks)))

        chunk_scores = self.score_batch(all_chunks)
        chunk_embeds = self.embed_batch(all_chunks)

        results: list[dict] = []
        for start, end in spans:
            w = weights[start:end]
            agg_probs = weighted_average([chunk_scores[i]["probs"] for i in range(start, end)], w)
            agg_embed = weighted_average([chunk_embeds[i] for i in range(start, end)], w)
            score, label = self._probs_to_score_label(agg_probs)
            results.append(
                {
                    "sentiment_score": score,
                    "sentiment_label": label,
                    "probs": agg_probs.tolist(),
                    "embedding": agg_embed.tolist(),
                    "n_chunks": end - start,
                }
            )
        return results

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return CLS-token embeddings (dim=768) for pgvector storage."""
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), settings.nlp_batch_size):
            batch = texts[i : i + settings.nlp_batch_size]
            inputs = self.tokenizer(
                batch, return_tensors="pt", truncation=True, padding=True, max_length=_MAX_TOKENS
            ).to(self.device)
            with torch.no_grad():
                hidden = self.model(**inputs, output_hidden_states=True).hidden_states[-1]
            cls = hidden[:, 0, :].float().cpu()
            embeddings.extend(cls.tolist())
        return embeddings

    def _ensure_fomc(self) -> None:
        """Lazily load FOMC-RoBERTa and resolve its hawkish/dovish class indices.

        The label order is resolved from the model config (with a documented
        fallback) and logged once, so a silent sign inversion is impossible to miss.
        """
        if hasattr(self, "_fomc_model"):
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._fomc_tokenizer = AutoTokenizer.from_pretrained(FOMC_ROBERTA)
        # Chunk manually with an explicit max_length, so silence the length check.
        self._fomc_tokenizer.model_max_length = int(1e9)
        fomc_model = AutoModelForSequenceClassification.from_pretrained(FOMC_ROBERTA)
        if self.use_fp16:
            fomc_model = fomc_model.half()
        self._fomc_model = fomc_model.to(self.device)
        self._fomc_model.eval()

        id2label = self._fomc_model.config.id2label
        self._fomc_hawk_idx, self._fomc_dove_idx = resolve_hawk_dove_indices(id2label)
        # Build the label map from the resolved indices (not a positional guess) so the
        # argmax label always agrees with the sign of hawkish_score.
        self._fomc_labels = {self._fomc_hawk_idx: "hawkish", self._fomc_dove_idx: "dovish"}
        for idx in range(len(id2label)):
            self._fomc_labels.setdefault(idx, "neutral")
        logger.info(
            f"FOMC-RoBERTa loaded: id2label={dict(id2label)} resolved "
            f"hawk_idx={self._fomc_hawk_idx}, dove_idx={self._fomc_dove_idx} "
            f"(hawkish_score = P[{self._fomc_hawk_idx}] - P[{self._fomc_dove_idx}])"
        )

    def _fomc_probs(self, texts: list[str]) -> list[list[float]]:
        """Raw softmax probabilities from FOMC-RoBERTa, one 512-token window per text."""
        self._ensure_fomc()
        out: list[list[float]] = []
        for i in range(0, len(texts), settings.nlp_batch_size):
            batch = texts[i : i + settings.nlp_batch_size]
            inputs = self._fomc_tokenizer(
                batch, return_tensors="pt", truncation=True, padding=True, max_length=_MAX_TOKENS
            ).to(self.device)
            with torch.no_grad():
                logits = self._fomc_model(**inputs).logits
            out.extend(logits.float().softmax(dim=-1).cpu().tolist())
        return out

    def _hawkish_from_probs(self, probs) -> tuple[float, str]:
        """Map a FOMC probability vector to (hawkish_score ∈ [-1, 1], label)."""
        probs = np.asarray(probs, dtype=float)
        score = float(probs[self._fomc_hawk_idx] - probs[self._fomc_dove_idx])
        label = self._fomc_labels[int(probs.argmax())]
        return score, label

    def score_hawkish_dovish(self, texts: list[str]) -> list[dict]:
        """Score full documents for hawkish/dovish stance with FOMC-RoBERTa.

        FOMC-RoBERTa truncates at 512 tokens, but central-bank speeches are far
        longer (median ~19k chars), so a single pass scores only the opening. This
        chunks each document into 512-token windows, scores every chunk, and
        token-weighted aggregates the class probabilities into one document result —
        the same full-document treatment as analyze_documents.

        Returns per-document dicts: hawkish_score ∈ [-1, 1], hawkish_label, probs
        (aggregated), n_chunks. Loads the model lazily on first call.
        """
        self._ensure_fomc()
        all_chunks: list[str] = []
        weights: list[int] = []
        spans: list[tuple[int, int]] = []
        for text in texts:
            chunks = self._chunk_text(text, tokenizer=self._fomc_tokenizer) or [("", 1)]
            start = len(all_chunks)
            for chunk_text, n_tokens in chunks:
                all_chunks.append(chunk_text)
                weights.append(n_tokens)
            spans.append((start, len(all_chunks)))

        chunk_probs = self._fomc_probs(all_chunks)

        results: list[dict] = []
        for start, end in spans:
            agg = weighted_average([chunk_probs[i] for i in range(start, end)], weights[start:end])
            score, label = self._hawkish_from_probs(agg)
            results.append(
                {
                    "hawkish_score": score,
                    "hawkish_label": label,
                    "probs": agg.tolist(),
                    "n_chunks": end - start,
                }
            )
        return results

    def agreement_score(self, stmt_embedding: list[float], rxn_embedding: list[float]) -> float:
        """Cosine similarity between two precomputed embeddings."""
        s = np.array(stmt_embedding, dtype=float)
        r = np.array(rxn_embedding, dtype=float)
        denom = np.linalg.norm(s) * np.linalg.norm(r)
        if denom == 0:
            return 0.0
        return float(np.dot(s, r) / denom)
