# SmartHireAI — Generative AI Resume Matching & AI Career Mentor

SmartHireAI is an end-to-end Generative AI career intelligence portal built for students and job seekers. It ingests PDF/DOCX resumes, extracts and validates structured candidate profiles using Google Gemini, performs semantic job matching using FAISS and sentence embeddings, produces targeted CV improvement suggestions, and provides a RAG-grounded Career Mentor chatbot.

---

## 1. System Architecture

```
   [ User Resume: PDF / DOCX ]
                │
                ▼
      ┌──────────────────┐
      │ Document Loader  │ (pypdf / python-docx + text sanitization)
      └─────────┬────────┘
                ▼
      ┌──────────────────┐
      │ LLM Parser       │ (Google Gemini structured JSON output)
      └─────────┬────────┘
                │
    ┌───────────┼──────────────────────────────┐
    ▼           ▼                              ▼
┌─────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│ Candidate│ │  Semantic Job Search │ │ CV Improvement Gen   │
│ Profile │ │  (FAISS + Embeddings)│ │ (Prompt Library +    │
│ (JSON)  │ └──────────┬───────────┘ │  Skill Gap & Critique)│
└─────────┘            ▼             └──────────────────────┘
            [ Ranked Job Matches ]
                       │
                       ▼
         ┌───────────────────────────┐
         │ AI Career Mentor (RAG)    │
         │ - Job Postings Corpus     │
         │ - Editable Career Notes   │
         │ - Guardrails Safety Layer │
         └─────────────┬─────────────┘
                       ▼
         ┌───────────────────────────┐
         │ Streamlit Web Portal UI   │
         └───────────────────────────┘
```

---

## 2. Key Modules & Technical Capabilities

1. **Resume Parser (`src/parsing/`):**
   - Extracts raw text from `.pdf`, `.docx`, and `.txt` documents.
   - Parses text using structured LLM output with Pydantic validation into five core fields: `name`, `skills`, `experience`, `education`, and `target_role`.
   - Handles missing, empty, or encrypted documents safely with informative UI feedback.
2. **Semantic Job Search (`src/search/`):**
   - Embeds job descriptions and candidate profiles using local `sentence-transformers` (`all-MiniLM-L6-v2`) by default, avoiding API embedding fees.
   - Vector indexing and top-$N$ cosine/L2 similarity search powered by **FAISS**.
   - Normalized relevance match scores between 0% and 100%.
3. **CV Improvement Generator (`src/generate/`):**
   - Curated prompt library with targeted prompt engineering.
   - Identifies specific missing skills when compared with a selected target job description.
   - Critiques weak, passive bullet points and provides truthful rewrites using the Google XYZ formula (*Accomplished [X] as measured by [Y], by doing [Z]*).
   - Generates a tailored professional summary labeled clearly as a reviewable draft.
4. **AI Career Mentor & RAG (`src/mentor/`):**
   - Grounded strictly in retrieved passages from the pre-collected job corpus and user-editable career notes (`data/career_notes/`).
   - Explicit hallucination safeguards: answers only from retrieved sources or states *"I don't have enough information in my career knowledge base to answer this."*
   - Returns citation metadata referencing the exact source documents used.
5. **Guardrails & Responsible AI (`src/safety/`):**
   - Pre-LLM safety filter preventing prompt injection, system instruction overrides, and out-of-scope requests (e.g., general chit-chat, medical/financial queries).
   - Sanitizes and isolates untrusted user resumes and chat messages.
6. **Interactive Streamlit Web Portal (`app/streamlit_app.py`):**
   - Tabbed, responsive UI covering resume upload, profile inspection, job recommendations, CV critique, and grounded career chat.

---

## 3. Project Directory Structure

```
SmartHireAI/
├── README.md                          # Project overview, setup, and architecture
├── requirements.txt                   # Production Python dependencies
├── .env.example                       # API key and environment template
├── .gitignore                         # Excludes secrets, large data, virtual envs
│
├── data/
│   ├── jobs/                          # Job dataset CSV (e.g., sample_jobs.csv)
│   ├── resumes/                       # Sample resumes for testing and evaluation
│   └── career_notes/                  # Curated markdown guides for RAG mentor
│
├── vectorstore/                       # Persisted FAISS index and metadata
│   ├── job_index.faiss
│   └── job_metadata.pkl
│
├── notebooks/
│   ├── 01_embeddings_explore.ipynb    # Embedding models & cosine similarity tests
│   ├── 02_build_faiss.ipynb           # FAISS index construction workflow
│   └── 03_rag_prototype.ipynb         # Interactive RAG mentor prototyping
│
├── src/
│   ├── config.py                      # Centralized constants, paths, and env config
│   ├── parsing/
│   │   ├── loader.py                  # Multi-format document loader (PDF, DOCX, TXT)
│   │   └── resume_parser.py           # Pydantic schemas and LLM structured parser
│   ├── search/
│   │   ├── embed.py                   # Embedding model wrapper (local & API options)
│   │   └── job_search.py              # FAISS indexing and top-N retrieval
│   ├── generate/
│   │   ├── prompts.py                 # Structured prompt library
│   │   └── cv_suggestions.py          # Skill gap and bullet point critique engine
│   ├── mentor/
│   │   └── rag_chain.py               # Grounded LangChain RAG pipeline
│   ├── safety/
│   │   └── guardrails.py              # Input validation and scope filtering
│   └── evaluate.py                    # Capstone evaluation scripts (hit rate, grounding)
│
├── app/
│   └── streamlit_app.py               # Complete Streamlit UI
│
└── reports/
    ├── answer_quality.md              # Evaluation report and prompt comparison
    └── final_report_draft.md          # Capstone project report documentation
```

---

## 4. Local Setup & Installation

### Step 1: Clone the Repository & Open the Directory
```powershell
cd C:\Users\lenovo\SmartHireAI
```

### Step 2: Create and Activate a Virtual Environment
```powershell
# Using Python 3.12
& "C:\Users\lenovo\AppData\Local\Programs\Python\Python312\python.exe" -m venv venv

# Activate on Windows PowerShell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Edit `.env` and insert your Google Gemini API key:
```ini
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```
> **Note:** Never commit `.env` or share your API key.

---

## 5. Usage & Running the Application

### 1. Build or Rebuild the FAISS Job Index
Before running the portal for the first time, build the FAISS index from the job dataset:
```powershell
python -m src.search.job_search --build
```

### 2. Launch the Streamlit Portal
```powershell
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

### 3. Run Capstone Evaluation Benchmarks
```powershell
python -m src.evaluate
```

---

## 6. Datasets & Knowledge Base Notes

- **Job Dataset (`data/jobs/`):** Includes a starter sample `sample_jobs.csv` containing common tech roles with fields: `title`, `company`, `skills`, `job_description`, `experience_level`, and `location`. You can replace this with any pre-collected Kaggle CSV by placing it in `data/jobs/` and configuring `JOBS_DATASET_PATH` in `.env`.
- **Career Notes (`data/career_notes/`):** Contains editable role guides (`data_analyst_guide.md`, `software_engineer_guide.md`, `interview_prep_and_transition.md`). These are loaded dynamically by the RAG mentor pipeline.

---

## 7. Responsible AI & Guardrails Policy

- **Input Validation:** Every prompt submitted to the LLM passes through `src/safety/guardrails.py` to prevent system prompt injection and out-of-domain abuse.
- **Strict Grounding:** The Career Mentor is constrained to answer only from retrieved context passages. If a user asks about medical advice, general trivia, or unmentioned topics, the mentor politely declines.
- **Data Privacy:** Resumes uploaded in the UI are held in memory during the session and are never transmitted to external services other than the user-configured LLM provider.

