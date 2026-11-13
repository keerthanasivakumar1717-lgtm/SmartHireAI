"""
SmartHireAI - Semantic Job Search
Indexes job descriptions using FAISS vector store and performs top-N cosine similarity queries
against candidate profiles.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import pickle
import csv
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

from src import config
from src.search.embed import generate_embedding, generate_embeddings_batch


class JobSearchIndex:
    """FAISS-backed semantic search index for job postings."""

    def __init__(self, index_path: Path = None, metadata_path: Path = None):
        self.index_path = index_path or config.FAISS_INDEX_FILE
        self.metadata_path = metadata_path or config.FAISS_METADATA_FILE
        self.index = None
        self.jobs_data: Optional[pd.DataFrame] = None
        self.is_faiss_available = False

        try:
            import faiss
            self.faiss = faiss
            self.is_faiss_available = True
        except ImportError:
            self.faiss = None
            self.is_faiss_available = False

    def load_and_normalize_jobs_csv(self, csv_path: Path = None) -> List[Dict[str, Any]]:
        """Load CSV and normalize column names using aliases. Returns list of record dicts."""
        path = csv_path or config.DEFAULT_JOBS_CSV
        if not path.exists():
            raise FileNotFoundError(f"Jobs dataset CSV not found at: {path}")

        records = []
        if pd is not None:
            df = pd.read_csv(path)
            renamed = {}
            for col in df.columns:
                clean_col = col.strip().lower()
                renamed[col] = config.COLUMN_ALIASES.get(clean_col, clean_col)
            df = df.rename(columns=renamed)
            for req in config.REQUIRED_JOB_COLUMNS:
                if req not in df.columns:
                    raise ValueError(f"Job dataset missing required column: {req}")
            for col in ["title", "company", "skills", "job_description"]:
                df[col] = df[col].fillna("").astype(str).str.strip()
            if "job_id" not in df.columns:
                df["job_id"] = [f"JOB_{i+1:04d}" for i in range(len(df))]
            records = df.to_dict(orient="records")
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                header_map = {}
                for h in reader.fieldnames or []:
                    clean_h = h.strip().lower()
                    header_map[h] = config.COLUMN_ALIASES.get(clean_h, clean_h)

                for i, row in enumerate(reader):
                    clean_row = {}
                    for orig_k, v in row.items():
                        clean_k = header_map.get(orig_k, orig_k)
                        clean_row[clean_k] = (v or "").strip()
                    if "job_id" not in clean_row or not clean_row["job_id"]:
                        clean_row["job_id"] = f"JOB_{i+1:04d}"
                    records.append(clean_row)

            if records:
                sample = records[0]
                for req in config.REQUIRED_JOB_COLUMNS:
                    if req not in sample:
                        raise ValueError(f"Job dataset missing required column: {req}")

        return records

    def build_index(self, csv_path: Path = None) -> int:
        """
        Embed job descriptions and build FAISS index.
        Saves index and metadata to disk.
        """
        records = self.load_and_normalize_jobs_csv(csv_path)
        if not records:
            raise ValueError("Cannot build index: jobs dataset contains 0 records.")

        # Construct rich semantic text representation for each job
        search_texts = []
        for r in records:
            text = (
                f"Title: {r.get('title', '')}. "
                f"Company: {r.get('company', '')}. "
                f"Key Required Skills: {r.get('skills', '')}. "
                f"Job Description: {r.get('job_description', '')}"
            )
            search_texts.append(text)

        # Generate embeddings (N x D)
        embeddings = generate_embeddings_batch(search_texts)
        dimension = embeddings.shape[1]

        # Ensure vectorstore directory exists
        config.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

        if self.is_faiss_available:
            faiss_index = self.faiss.IndexFlatIP(dimension)
            faiss_index.add(embeddings)
            self.faiss.write_index(faiss_index, str(self.index_path))
            self.index = faiss_index
        else:
            npy_path = self.index_path.with_suffix(".npy")
            np.save(str(npy_path), embeddings)
            self.index = embeddings

        # Save metadata records
        with open(self.metadata_path, "wb") as f:
            pickle.dump(records, f)

        self.jobs_data = records
        return len(records)

    def load_index(self) -> bool:
        """Load saved FAISS index and metadata from disk."""
        if not self.metadata_path.exists():
            return False

        with open(self.metadata_path, "rb") as f:
            self.jobs_data = pickle.load(f)

        if self.is_faiss_available and self.index_path.exists():
            self.index = self.faiss.read_index(str(self.index_path))
            return True
        elif self.index_path.with_suffix(".npy").exists():
            self.index = np.load(str(self.index_path.with_suffix(".npy")))
            return True

        return False

    def is_ready(self) -> bool:
        """Check if index and metadata are loaded in memory."""
        return self.index is not None and self.jobs_data is not None and len(self.jobs_data) > 0

    def search(
        self,
        query_text: str,
        candidate_skills: Optional[List[str]] = None,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for top-K matching jobs given query text and candidate skills.
        Returns a list of dicts with match scores and skill overlap.
        """
        if not self.is_ready():
            loaded = self.load_index()
            if not loaded:
                # Try auto-building from default dataset
                if config.DEFAULT_JOBS_CSV.exists():
                    self.build_index()
                else:
                    raise RuntimeError("FAISS index is not built and no job dataset is present.")

        query_vec = generate_embedding(query_text)
        candidate_skills_set = {s.lower().strip() for s in (candidate_skills or [])}

        results = []

        if self.is_faiss_available and hasattr(self.index, "search"):
            # FAISS search
            query_matrix = np.expand_dims(query_vec, axis=0)
            scores, indices = self.index.search(query_matrix, min(top_k * 2, len(self.jobs_data)))
            raw_scores = scores[0]
            matched_indices = indices[0]
        else:
            # NumPy matrix dot product fallback
            matrix = self.index  # N x D
            raw_scores = np.dot(matrix, query_vec)
            matched_indices = np.argsort(raw_scores)[::-1][: min(top_k * 2, len(self.jobs_data))]
            raw_scores = raw_scores[matched_indices]

        for score, idx in zip(raw_scores, matched_indices):
            if idx < 0 or idx >= len(self.jobs_data):
                continue

            # Convert cosine score (-1 to 1) to percentage (0% to 100%)
            norm_score = max(0.0, min(1.0, (float(score) + 1.0) / 2.0))
            if norm_score < threshold:
                continue

            if hasattr(self.jobs_data, "iloc"):
                row = self.jobs_data.iloc[idx].to_dict()
            else:
                row = dict(self.jobs_data[idx])

            # Analyze skill overlap
            job_skills_raw = str(row.get("skills", ""))
            job_skills = [
                s.strip() for s in job_skills_raw.replace(";", ",").split(",") if s.strip()
            ]

            matched = []
            missing = []
            for js in job_skills:
                if any(cs in js.lower() or js.lower() in cs for cs in candidate_skills_set):
                    matched.append(js)
                else:
                    missing.append(js)

            row["similarity_score"] = float(score)
            row["match_percentage"] = round(norm_score * 100, 1)
            row["matched_skills"] = matched
            row["missing_skills"] = missing
            results.append(row)

            if len(results) >= top_k:
                break

        return results


def get_default_job_searcher() -> JobSearchIndex:
    """Convenience factory function."""
    searcher = JobSearchIndex()
    if not searcher.is_ready():
        searcher.load_index()
    return searcher


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SmartHireAI FAISS Index Builder")
    parser.add_argument("--build", action="store_true", help="Build or rebuild the job index from CSV")
    parser.add_argument("--csv", type=str, default=str(config.DEFAULT_JOBS_CSV), help="Path to jobs CSV")
    parser.add_argument("--query", type=str, default="Junior Data Analyst with SQL and Python skills", help="Test query")

    args = parser.parse_args()

    searcher = JobSearchIndex()
    if args.build:
        print(f"Building FAISS job index from {args.csv}...")
        count = searcher.build_index(Path(args.csv))
        print(f"Success! Indexed {count} job descriptions into {config.VECTORSTORE_DIR}")

    print(f"\nRunning test search for query: '{args.query}'...")
    results = searcher.search(args.query, top_k=3)
    for i, res in enumerate(results, 1):
        print(f"\n[{i}] {res['title']} @ {res['company']} (Match: {res['match_percentage']}%)")
        print(f"    Required Skills: {res['skills']}")
        print(f"    Matched Skills: {res['matched_skills']}")
        print(f"    Missing Skills: {res['missing_skills']}")
