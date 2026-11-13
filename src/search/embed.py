"""
SmartHireAI - Embedding Generator
Generates high-dimensional semantic vector embeddings using local sentence-transformers,
with optional Google Gemini cloud embeddings and a deterministic TF-IDF fallback.
"""

from typing import List, Optional
import numpy as np

from src import config

# Global cached embedding model instance
_CACHED_LOCAL_MODEL = None


def _get_local_sentence_transformer(model_name: Optional[str] = None):
    """Lazy load and cache sentence-transformers model."""
    global _CACHED_LOCAL_MODEL
    if _CACHED_LOCAL_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            name = model_name or config.DEFAULT_EMBEDDING_MODEL
            _CACHED_LOCAL_MODEL = SentenceTransformer(name, device=config.EMBEDDING_DEVICE)
        except ImportError:
            _CACHED_LOCAL_MODEL = "UNAVAILABLE"
        except Exception as e:
            print(f"[SmartHireAI] Notice: Could not load sentence-transformers ({e}). Using fallback vectorizer.")
            _CACHED_LOCAL_MODEL = "UNAVAILABLE"
    return _CACHED_LOCAL_MODEL


def _deterministic_fallback_embedding(text: str, dim: int = 384) -> np.ndarray:
    """
    Deterministic normalized vectorizer used when sentence-transformers is not installed.
    Computes a deterministic hashed feature vector with L2 normalization.
    """
    if not text:
        vec = np.zeros(dim, dtype=np.float32)
        vec[0] = 1.0
        return vec

    # Tokenize words
    words = [w.lower().strip() for w in text.split() if w.strip()]
    vec = np.zeros(dim, dtype=np.float32)

    for word in words:
        # Simple polynomial rolling hash
        h = 0
        for char in word:
            h = (h * 31 + ord(char)) & 0xFFFFFFFF
        idx = h % dim
        # Assign frequency weight
        vec[idx] += 1.0

    # L2 normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    else:
        vec[0] = 1.0
    return vec


def generate_embedding(text: str, model_name: Optional[str] = None) -> np.ndarray:
    """
    Generate a 1D float32 numpy embedding vector for a single text string.
    """
    local_st = _get_local_sentence_transformer(model_name)
    if local_st != "UNAVAILABLE":
        emb = local_st.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return emb.astype(np.float32)

    # Check for Gemini embedding if key is available
    if config.is_gemini_configured() and config.DEFAULT_EMBEDDING_MODEL.startswith("models/"):
        try:
            import google.generativeai as genai
            genai.configure(api_key=config.GEMINI_API_KEY)
            result = genai.embed_content(
                model=config.DEFAULT_EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_document"
            )
            emb = np.array(result["embedding"], dtype=np.float32)
            norm = np.linalg.norm(emb)
            return (emb / norm) if norm > 0 else emb
        except Exception:
            pass

    return _deterministic_fallback_embedding(text, dim=384)


def generate_embeddings_batch(texts: List[str], model_name: Optional[str] = None) -> np.ndarray:
    """
    Generate a 2D float32 numpy array (N x D) for a list of text strings.
    """
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    local_st = _get_local_sentence_transformer(model_name)
    if local_st != "UNAVAILABLE":
        embs = local_st.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32
        )
        return embs.astype(np.float32)

    # Fallback to row-by-row deterministic embedding
    vectors = [_deterministic_fallback_embedding(t, dim=384) for t in texts]
    return np.vstack(vectors).astype(np.float32)

