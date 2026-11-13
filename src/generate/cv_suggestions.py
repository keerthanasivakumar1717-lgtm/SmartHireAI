"""
SmartHireAI - CV Improvement Generator
Generates actionable resume feedback, skill gap analysis, bullet point rewrites,
and tailored professional summaries.
"""

from typing import Dict, Any, List
import re

from src import config
from src.parsing.resume_parser import ParsedResume
from src.generate.prompts import (
    SYSTEM_CV_IMPROVEMENT_PROMPT,
    PROMPT_UNIFIED_CV_IMPROVEMENT,
)


def _extract_all_bullets(profile: ParsedResume) -> List[str]:
    """Collect all bullet points across candidate experience items."""
    bullets = []
    for exp in profile.experience:
        for b in exp.bullets:
            if b.strip():
                bullets.append(b.strip())
    return bullets


def generate_offline_cv_suggestions(profile: ParsedResume, target_job: Dict[str, Any]) -> str:
    """
    Deterministic rule-based generator used when offline or without Gemini API key.
    Provides rigorous skill gap calculation and XYZ bullet improvements.
    """
    candidate_skills_set = {s.lower().strip() for s in profile.skills}
    job_skills_raw = str(target_job.get("skills", ""))
    job_skills_list = [s.strip() for s in job_skills_raw.replace(";", ",").split(",") if s.strip()]

    matched = []
    missing = []
    for js in job_skills_list:
        if any(cs in js.lower() or js.lower() in cs for cs in candidate_skills_set):
            matched.append(js)
        else:
            missing.append(js)

    job_title = target_job.get("title", "Target Role")
    job_company = target_job.get("company", "Hiring Organization")

    # Analyze bullets
    candidate_bullets = _extract_all_bullets(profile)
    critique_items = []

    weak_verbs = ["worked on", "assisted in", "helped with", "responsible for", "participated in", "handled"]

    if candidate_bullets:
        for b in candidate_bullets[:3]:
            is_weak = any(wv in b.lower() for wv in weak_verbs)
            critique = (
                "Uses passive phrasing and lacks measurable business impact."
                if is_weak
                else "Good start, but could be elevated with clear metrics and technical methodology."
            )
            # Create XYZ rewrite
            cleaned_b = re.sub(r"^(worked on|assisted in|helped with|responsible for)\s*", "", b, flags=re.IGNORECASE)
            rewritten = (
                f"Spearheaded {cleaned_b.lower()}, achieving [quantifiable metric, e.g. 20% efficiency gain] "
                f"by implementing industry-standard {job_skills_list[0] if job_skills_list else 'methodologies'}."
            )
            critique_items.append((b, critique, rewritten))
    else:
        critique_items.append((
            "No specific project bullet points were detected in the resume.",
            "Resumes without clear project achievements fail ATS screening.",
            f"Engineered and deployed a {job_title} feature, reducing [metric] by [X%] using {', '.join(job_skills_list[:2])}."
        ))

    # Draft Summary
    skills_preview = ", ".join(profile.skills[:4]) if profile.skills else "Modern computing frameworks"
    draft_summary = (
        f"Results-oriented {profile.target_role} with proven experience in {skills_preview}. "
        f"Demonstrated track record of delivering end-to-end technical solutions and collaborating in cross-functional environments. "
        f"Seeking to contribute technical depth and continuous improvement as a {job_title} at {job_company}."
    )

    matched_md = ", ".join(f"`{m}`" for m in matched) if matched else "*None directly identified from primary skills list.*"
    missing_md = ", ".join(f"`{m}`" for m in missing) if missing else "*None detected — strong skill alignment!*"

    bullets_md = ""
    for orig, crit, imp in critique_items:
        bullets_md += f"""
- **Original:** "{orig}"
  - **Critique:** {crit}
  - **Improved Draft (XYZ Formula):** *"{imp}"*
"""

    return f"""### CV Improvement Recommendations (Offline Analysis Mode)

> **Draft Notice:** The following suggestions are structured recommendations. Always verify that all numbers and achievements truthfully reflect your actual experience.

---

## 1. Skill Gap Analysis for: **{job_title}** at **{job_company}**

- **Directly Matched Skills:** {matched_md}
- **Critical Missing Skills to Add/Learn:** {missing_md}

**Bridging Recommendation:**
Review the missing skills listed above. If you have worked with these technologies in coursework or personal projects, make sure they appear explicitly on your CV. If not, consider building a targeted mini-project demonstrating {missing[0] if missing else 'advanced domain skills'}.

---

## 2. Bullet Point Critiques & Truthful Rewrites
{bullets_md}

---

## 3. Tailored Professional Summary Draft
> *"{draft_summary}"*

*(Draft suggestion: Review and adjust wording to ensure complete alignment with your personal background.)*

---

## 4. Key Takeaways & Immediate Next Steps
1. Add explicit keywords for `{missing[0] if missing else 'job requirements'}` in your Technical Skills section.
2. Upgrade at least 3 resume bullet points to follow the Google XYZ formula: **Accomplished [X] as measured by [Y], by doing [Z]**.
"""


def generate_cv_improvements(
    profile: ParsedResume,
    target_job: Dict[str, Any],
    prefer_offline: bool = False
) -> str:
    """
    Generate comprehensive CV improvement suggestions comparing candidate profile with target job.
    Uses Google Gemini if configured, with graceful fallback to offline heuristic analysis.
    """
    if not prefer_offline and config.is_gemini_configured():
        try:
            import google.generativeai as genai

            genai.configure(api_key=config.GEMINI_API_KEY)
            model = genai.GenerativeModel(
                model_name=config.DEFAULT_LLM_MODEL,
                system_instruction=SYSTEM_CV_IMPROVEMENT_PROMPT,
                generation_config={"temperature": 0.2},
            )

            candidate_bullets = _extract_all_bullets(profile)
            bullets_str = "\n".join(f"- {b}" for b in candidate_bullets) if candidate_bullets else "None provided."

            prompt = PROMPT_UNIFIED_CV_IMPROVEMENT.format(
                name=profile.name,
                target_role=profile.target_role,
                candidate_skills=", ".join(profile.skills),
                candidate_summary=profile.summary or "Not specified",
                candidate_bullets=bullets_str,
                job_title=target_job.get("title", "Target Role"),
                job_company=target_job.get("company", "Company"),
                job_skills=target_job.get("skills", "Technical skills"),
                job_description=target_job.get("job_description", "Job requirements"),
            )

            response = model.generate_content(prompt)
            output = response.text.strip()

            header = (
                "> **AI-Generated Review Draft:** Review all suggestions for personal truthfulness. "
                "Never include fabricated metrics or qualifications.\n\n"
            )
            return header + output
        except Exception as e:
            print(f"[SmartHireAI] Gemini CV suggestion failed ({e}). Falling back to offline generator.")

    return generate_offline_cv_suggestions(profile, target_job)

