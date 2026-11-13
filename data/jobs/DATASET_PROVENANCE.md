# Dataset Provenance & Data Notice

## Status of `data/jobs/sample_jobs.csv`
> **DATASET TYPE: SYNTHETIC SAMPLE DATA (DEMONSTRATION ONLY)**

The file `data/jobs/sample_jobs.csv` contains **curated synthetic sample data** created solely for local schema validation, test-profile hit-rate benchmarks, and demonstration of the semantic search pipeline. 

- **Not Real Listings:** The companies (e.g., *Apex Analytics*, *CloudScale Technologies*, *NeuralCraft Labs*) and descriptions are realistic mock records. They are **not** real-world active job vacancies.
- **No Live Scraping:** In strict compliance with the capstone instructions and Terms of Service, **no web scraping of LinkedIn, Naukri, or any job portal was conducted**.

---

## Using a Real Kaggle Dataset

To replace this synthetic sample with a real pre-collected dataset:

1. Download a pre-collected, sanctioned job posting dataset from Kaggle (e.g., *Naukri Job Postings* or *LinkedIn Job Postings* CSV).
2. Place the CSV file into this directory: `data/jobs/kaggle_jobs.csv`.
3. Ensure the CSV contains (or maps via `COLUMN_ALIASES` in `src/config.py` to) the required fields:
   - `title`
   - `company`
   - `skills`
   - `job_description`
4. Rebuild the FAISS index by running:
   ```powershell
   python -m src.search.job_search --build --csv data/jobs/kaggle_jobs.csv
   ```
5. Update your `.env` file:
   ```ini
   JOBS_DATASET_PATH=data/jobs/kaggle_jobs.csv
   ```
