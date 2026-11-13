"""
SmartHireAI - Prompt Library
Curated, version-controlled prompt templates for CV analysis, skill gaps, bullet rewrites,
and tailored professional summaries.
"""

from typing import Dict, Any

# System Prompt for CV Improvement Engine
SYSTEM_CV_IMPROVEMENT_PROMPT = """You are an elite Technical Career Advisor and Resume Strategist.
Your mission is to provide high-impact, truthful, and actionable resume optimization recommendations
for a candidate applying to a specific target job posting.

NON-NEGOTIABLE PRINCIPLES:
1. TRUTHFULNESS: Never invent false metrics, imaginary certifications, fabricated employment dates,
   or experiences the candidate did not have.
2. XYZ FORMULA: For bullet points, use Google's XYZ formula:
   'Accomplished [X] as measured by [Y], by doing [Z]' using placeholder metrics (e.g. '[X% increase]')
   if exact figures were omitted by the candidate, advising them to supply their real numbers.
3. SPECIFICITY: Avoid vague advice like 'learn more skills'. Identify concrete technologies, libraries,
   and concepts from the job description.
4. REVIEWABLE DRAFTS: Clearly label all rewritten text as drafts for the candidate's review and validation.
"""

# Prompt for Skill Gap Analysis
PROMPT_SKILL_GAP = """### TASK: Skill Gap Analysis
Compare the candidate's profile against the target job requirements.

CANDIDATE PROFILE:
- Target Role: {target_role}
- Current Skills: {candidate_skills}
- Experience Highlights: {candidate_experience_summary}

TARGET JOB POSTING:
- Title: {job_title}
- Company: {job_company}
- Required Skills: {job_skills}
- Job Description: {job_description}

INSTRUCTIONS:
1. Identify high-priority missing technical skills present in the job description but absent from the candidate's resume.
2. Identify matched skills where the candidate is already well-aligned.
3. Suggest 2-3 specific learning resources or project types to bridge the top missing skill gaps.
"""

# Prompt for Bullet Point Critique and Truthful Rewrite
PROMPT_BULLET_REWRITE = """### TASK: Bullet Point Critique & Truthful Rewrite
Critique weak or passive resume bullet points and provide stronger, impact-oriented rewrites.

CANDIDATE RESUME BULLETS:
{candidate_bullets}

TARGET JOB CONTEXT:
- Role: {job_title}
- Required Technologies: {job_skills}

INSTRUCTIONS:
1. Select 2-3 of the weakest or most passive bullets from the candidate's list (e.g., bullets starting with 'Worked on', 'Assisted with', 'Responsible for').
2. Provide a constructive critique explaining why the original bullet is weak (e.g., lacks quantifiable impact, passive voice, missing tools).
3. Provide a rewritten version using strong active verbs and the XYZ formula: 'Accomplished [X], measured by [Y], by doing [Z]'.
4. Do NOT fabricate numbers. If a metric is needed, use bracketed guidance like '[e.g., reduced loading latency by X%]'.
"""

# Prompt for Tailored Summary Generation
PROMPT_TAILORED_SUMMARY = """### TASK: Tailored Professional Summary Draft
Draft an impactful 3-sentence professional summary tailored for the target role.

CANDIDATE BACKGROUND:
- Stated Target Role: {target_role}
- Current Summary: {candidate_summary}
- Confirmed Skills: {candidate_skills}
- Years / Level of Experience: {candidate_experience_summary}

TARGET JOB CONTEXT:
- Job Title: {job_title}
- Company: {job_company}
- Key Stack: {job_skills}

INSTRUCTIONS:
1. Sentence 1: Hook stating candidate title, relevant experience level, and core technical domain.
2. Sentence 2: Key technical competencies directly aligning with the target job's stack.
3. Sentence 3: Target contribution and value the candidate seeks to bring to {job_company}.
4. Note: Clearly emphasize that this is a suggested draft to be checked for accuracy.
"""

# Comprehensive Unified Prompt for single-turn complete feedback
PROMPT_UNIFIED_CV_IMPROVEMENT = """You are analyzing a candidate's resume against a target job.

=== CANDIDATE PROFILE ===
Name: {name}
Target Role: {target_role}
Skills: {candidate_skills}
Summary: {candidate_summary}
Experience Bullets:
{candidate_bullets}

=== TARGET JOB ===
Title: {job_title}
Company: {job_company}
Required Skills: {job_skills}
Description:
{job_description}

=== OUTPUT FORMAT ===
Produce a structured markdown report with the following four sections:

## 1. Skill Gap Analysis
- **Directly Matched Skills:** List skills the candidate has that match the job.
- **Critical Missing Skills:** List required skills from the job posting missing from the resume.
- **Bridge Recommendations:** Actionable advice on projects or topics to bridge the top 2 gaps.

## 2. Bullet Point Critiques & Truthful Rewrites
For at least 2 bullets:
- **Original:** "[Original text]"
- **Critique:** Why it is passive or lacking impact.
- **Improved Draft:** "[Active XYZ rewrite - do not invent false facts, use brackets for missing metrics]"

## 3. Tailored Professional Summary Draft
> "[Draft 3-sentence summary tailored to {job_title} at {job_company}]"
*(Draft suggestion: verify all claims before using)*

## 4. Key Takeaway & Next Action
One concrete next step for the candidate.
"""

# Registry of prompt templates for evaluation and prompt comparison benchmarks
PROMPT_LIBRARY: Dict[str, Dict[str, Any]] = {
    "v1_baseline_prompt": {
        "description": "Baseline naive prompt without XYZ constraints or anti-hallucination rules",
        "template": "Give suggestions to improve this resume for this job:\nResume: {candidate_skills}\nJob: {job_description}",
    },
    "v2_production_unified": {
        "description": "Production structured prompt with XYZ guidelines and anti-fabrication rules",
        "system": SYSTEM_CV_IMPROVEMENT_PROMPT,
        "template": PROMPT_UNIFIED_CV_IMPROVEMENT,
    },
    "skill_gap": {
        "description": "Targeted skill gap comparative prompt",
        "template": PROMPT_SKILL_GAP,
    },
    "bullet_rewrite": {
        "description": "XYZ bullet point critique and rewrite prompt",
        "template": PROMPT_BULLET_REWRITE,
    },
    "tailored_summary": {
        "description": "3-sentence tailored professional summary prompt",
        "template": PROMPT_TAILORED_SUMMARY,
    },
}
