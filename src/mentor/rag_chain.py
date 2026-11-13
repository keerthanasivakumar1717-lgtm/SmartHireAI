"""
SmartHireAI - AI Career Mentor RAG Pipeline
Retrieval-Augmented Generation chatbot grounded strictly in the job corpus and career notes,
orchestrated with LangChain principles and strict anti-hallucination guardrails.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple
import re

from src import config
from src.search.embed import generate_embedding, generate_embeddings_batch
from src.safety.guardrails import validate_user_query, sanitize_input

SYSTEM_MENTOR_GROUNDING_PROMPT = """You are the SmartHire AI Career Mentor.
Your role is to guide students and job seekers using STRICTLY the provided career knowledge base and job postings below.

STRICT GROUNDING RULES:
1. ONLY answer based on the facts and information contained in the 'RETRIEVED CONTEXT' section.
2. If the context does not contain enough information to answer the question, you MUST reply:
   "I do not have enough information in my career knowledge base to answer this question. Please refer to specialized resources or consider adding more notes to the career knowledge base."
3. Do NOT make up facts, statistics, prerequisites, company policies, or personal opinions outside the context.
4. Always cite which document or guide you drew the answer from at the end of your response.
5. Keep your tone encouraging, professional, and directly actionable.
"""


class KnowledgeChunk:
    """Represents a chunked document passage with provenance metadata."""

    def __init__(self, content: str, source_name: str, doc_type: str, section: str = ""):
        self.content = content.strip()
        self.source_name = source_name
        self.doc_type = doc_type  # 'career_note' or 'job_posting'
        self.section = section

    def __repr__(self):
        return f"<KnowledgeChunk src='{self.source_name}' len={len(self.content)}>"


class CareerMentorRAG:
    """RAG pipeline managing knowledge ingestion, vector retrieval, and grounded response synthesis."""

    def __init__(self):
        self.chunks: List[KnowledgeChunk] = []
        self.chunk_embeddings = None
        self._is_initialized = False

    def _chunk_text(self, text: str, source: str, doc_type: str, chunk_size: int = 150) -> List[KnowledgeChunk]:
        """Split text into manageable chunks respecting markdown headings."""
        sections = re.split(r"\n(?=#{1,3}\s)", text)
        result = []

        for sec in sections:
            lines = sec.strip().splitlines()
            if not lines:
                continue
            title = lines[0].replace("#", "").strip() if lines[0].startswith("#") else "General Guidance"
            words = sec.split()

            # Split large sections into word-based windows
            for i in range(0, max(1, len(words)), chunk_size):
                window = " ".join(words[i : i + chunk_size + 30])
                if len(window.strip()) > 40:
                    result.append(KnowledgeChunk(
                        content=window.strip(),
                        source_name=source,
                        doc_type=doc_type,
                        section=title
                    ))
        return result

    def load_knowledge_base(self) -> int:
        """Load career notes and job postings from data directory."""
        all_chunks: List[KnowledgeChunk] = []

        # 1. Load Career Notes from data/career_notes/
        notes_dir = config.CAREER_NOTES_DIR
        if notes_dir.exists():
            for note_path in notes_dir.glob("*.md"):
                try:
                    with open(note_path, "r", encoding="utf-8", errors="replace") as f:
                        text = f.read()
                    chunks = self._chunk_text(text, source=note_path.name, doc_type="career_note")
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"[SmartHireAI] Error reading note {note_path}: {e}")

        # 2. Load Job Descriptions from data/jobs/
        if config.DEFAULT_JOBS_CSV.exists():
            try:
                import csv
                with open(config.DEFAULT_JOBS_CSV, "r", encoding="utf-8", errors="replace") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        title = row.get("title", "")
                        comp = row.get("company", "")
                        desc = row.get("job_description", "")
                        skills = row.get("skills", "")
                        content = f"Job Posting: {title} at {comp}. Required Skills: {skills}. Overview: {desc}"
                        all_chunks.append(KnowledgeChunk(
                            content=content,
                            source_name=f"Job: {title} ({comp})",
                            doc_type="job_posting",
                            section=title
                        ))
            except Exception as e:
                print(f"[SmartHireAI] Error loading jobs into RAG: {e}")

        self.chunks = all_chunks
        if all_chunks:
            texts = [c.content for c in all_chunks]
            self.chunk_embeddings = generate_embeddings_batch(texts)
            self._is_initialized = True

        return len(all_chunks)

    def retrieve(self, query: str, top_k: int = 4) -> List[Tuple[KnowledgeChunk, float]]:
        """Retrieve the top-K most semantically relevant chunks for a user query."""
        if not self._is_initialized or not self.chunks:
            self.load_knowledge_base()

        if not self.chunks:
            return []

        query_vec = generate_embedding(query)
        # Compute cosine similarity
        scores = self.chunk_embeddings.dot(query_vec)
        top_indices = scores.argsort()[::-1][:top_k]

        retrieved = []
        for idx in top_indices:
            retrieved.append((self.chunks[idx], float(scores[idx])))
        return retrieved

    def answer_question(
        self,
        query: str,
        chat_history: List[Dict[str, str]] = None,
        prefer_offline: bool = False
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline:
        1. Guardrail validation
        2. Knowledge retrieval
        3. Grounded LLM response synthesis (or offline extractive synthesis)
        4. Citation generation
        """
        # Step 1: Guardrails check
        is_safe, refusal = validate_user_query(query)
        if not is_safe:
            return {
                "answer": refusal,
                "sources": [],
                "grounded": False,
                "guardrail_triggered": True,
            }

        # Step 2: Retrieve relevant passages
        retrieved_items = self.retrieve(query, top_k=3)
        if not retrieved_items:
            return {
                "answer": (
                    "I do not have enough information in my career knowledge base to answer this question. "
                    "The knowledge base currently contains no documents."
                ),
                "sources": [],
                "grounded": False,
                "guardrail_triggered": False,
            }

        # Format context and citations
        context_blocks = []
        sources = []
        seen_sources = set()

        for chunk, score in retrieved_items:
            context_blocks.append(f"[{chunk.source_name} - {chunk.section}]\n{chunk.content}")
            if chunk.source_name not in seen_sources:
                sources.append({
                    "name": chunk.source_name,
                    "section": chunk.section,
                    "relevance_score": round(max(0.0, min(1.0, (score + 1.0) / 2.0)) * 100, 1),
                })
                seen_sources.add(chunk.source_name)

        context_str = "\n\n---\n\n".join(context_blocks)

        # Step 3: LLM Generation (if configured)
        if not prefer_offline and config.is_gemini_configured():
            try:
                import google.generativeai as genai

                genai.configure(api_key=config.GEMINI_API_KEY)
                model = genai.GenerativeModel(
                    model_name=config.DEFAULT_LLM_MODEL,
                    system_instruction=SYSTEM_MENTOR_GROUNDING_PROMPT,
                    generation_config={"temperature": 0.2},
                )

                # Format chat history if present
                history_str = ""
                if chat_history:
                    recent = chat_history[-4:]
                    history_str = "PREVIOUS CONVERSATION CONTEXT:\n" + "\n".join(
                        f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}"
                        for msg in recent
                    ) + "\n\n"

                prompt = (
                    f"{history_str}"
                    f"RETRIEVED CONTEXT PASSAGES:\n{context_str}\n\n"
                    f"USER QUESTION: {sanitize_input(query)}"
                )

                response = model.generate_content(prompt)
                answer_text = response.text.strip()

                return {
                    "answer": answer_text,
                    "sources": sources,
                    "grounded": True,
                    "guardrail_triggered": False,
                }
            except Exception as e:
                print(f"[SmartHireAI] Gemini RAG call encountered error: {e}. Using offline synthesis.")

        # Offline Grounded Synthesis
        # Check if query matches content reasonably (top score check)
        top_score = retrieved_items[0][1]
        norm_top_score = (top_score + 1.0) / 2.0

        if norm_top_score < 0.45:
            return {
                "answer": (
                    "I do not have enough information in my career knowledge base to answer this question. "
                    "The available role guides and job postings do not cover this topic in sufficient detail."
                ),
                "sources": sources,
                "grounded": False,
                "guardrail_triggered": False,
            }

        # Synthesize from top chunks
        primary_chunk = retrieved_items[0][0]
        secondary_chunk = retrieved_items[1][0] if len(retrieved_items) > 1 else None

        answer_body = (
            f"Based on **{primary_chunk.source_name}** ({primary_chunk.section}):\n\n"
            f"> {primary_chunk.content}\n\n"
        )
        if secondary_chunk and secondary_chunk.source_name != primary_chunk.source_name:
            answer_body += (
                f"Additionally, related context from **{secondary_chunk.source_name}** states:\n\n"
                f"> {secondary_chunk.content}\n\n"
            )

        answer_body += (
            "*(Answer generated from the local knowledge base. "
            "For full conversational synthesis, configure GEMINI_API_KEY in .env.)*"
        )

        return {
            "answer": answer_body,
            "sources": sources,
            "grounded": True,
            "guardrail_triggered": False,
        }


# Global singleton instance
_GLOBAL_MENTOR = None


def get_career_mentor() -> CareerMentorRAG:
    """Factory function providing a cached CareerMentor instance."""
    global _GLOBAL_MENTOR
    if _GLOBAL_MENTOR is None:
        _GLOBAL_MENTOR = CareerMentorRAG()
        _GLOBAL_MENTOR.load_knowledge_base()
    return _GLOBAL_MENTOR
