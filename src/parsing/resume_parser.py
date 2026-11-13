"""
SmartHireAI - Resume Parser
Extracts structured, validated candidate profiles from resume text using Google Gemini
with robust Pydantic schemas and a deterministic offline heuristic fallback.
"""

from typing import List, Optional
import json
import re
from pydantic import BaseModel, Field

from src import config


class EducationItem(BaseModel):
    degree: str = Field(..., description="Degree or certification title")
    institution: Optional[str] = Field(None, description="University, college, or school name")
    year: Optional[str] = Field(None, description="Graduation year or date range")
    details: Optional[str] = Field(None, description="Additional academic highlights or honors")


class ExperienceItem(BaseModel):
    role: str = Field(..., description="Job title or role name")
    company: Optional[str] = Field(None, description="Company or organization name")
    duration: Optional[str] = Field(None, description="Duration or dates of employment")
    bullets: List[str] = Field(default_factory=list, description="Key responsibilities and achievements")


class ParsedResume(BaseModel):
    name: str = Field("Candidate", description="Full candidate name")
    email: Optional[str] = Field(None, description="Contact email address")
    phone: Optional[str] = Field(None, description="Contact phone number")
    location: Optional[str] = Field(None, description="City, State, or Country")
    target_role: str = Field("General Tech Role", description="Target role or primary specialization")
    summary: str = Field("", description="Professional summary or bio")
    skills: List[str] = Field(default_factory=list, description="List of technical and domain skills")
    experience: List[ExperienceItem] = Field(default_factory=list, description="Work experience and internships")
    education: List[EducationItem] = Field(default_factory=list, description="Academic qualifications")
    raw_text_preview: str = Field("", description="First 500 characters of raw resume text")
    parsing_method: str = Field("gemini", description="'gemini' or 'offline_heuristic'")

    def to_profile_summary_text(self) -> str:
        """Construct a semantic text representation of the profile for vector embedding."""
        skills_str = ", ".join(self.skills) if self.skills else "General technical skills"
        exp_summary = " ".join(
            f"{e.role} at {e.company or 'Company'}. {' '.join(e.bullets[:2])}"
            for e in self.experience
        )
        return (
            f"Target Role: {self.target_role}. "
            f"Summary: {self.summary}. "
            f"Key Skills: {skills_str}. "
            f"Experience: {exp_summary}"
        ).strip()


SYSTEM_RESUME_PARSER_PROMPT = """You are an expert AI Resume Parsing Engine.
Your task is to analyze the provided resume text and extract the candidate's structured profile into strict, valid JSON.

CRITICAL INSTRUCTIONS:
1. Output ONLY a valid JSON object matching the requested schema. No conversational filler, no markdown explanations outside the JSON.
2. Do NOT hallucinate or invent qualifications, metrics, companies, or dates. If a field is not present in the resume, leave it as null or an empty list.
3. Normalize skills into concise, recognizable industry keywords (e.g., 'Python', 'PostgreSQL', 'React', 'Docker').
4. If target_role is not explicitly written in the resume, infer the most accurate target role from the candidate's recent experience and technical skills.

Required JSON Structure:
{
  "name": "Full Name",
  "email": "email@example.com or null",
  "phone": "+1234567890 or null",
  "location": "City, Country or null",
  "target_role": "Target Job Title",
  "summary": "2-3 sentence professional summary based strictly on the resume",
  "skills": ["Skill1", "Skill2", "Skill3"],
  "experience": [
    {
      "role": "Job Title",
      "company": "Company Name",
      "duration": "Jan 2023 - Present",
      "bullets": ["Achievement 1", "Responsibility 2"]
    }
  ],
  "education": [
    {
      "degree": "Degree Name",
      "institution": "University Name",
      "year": "2024",
      "details": "CGPA / Honors if mentioned or null"
    }
  ]
}
"""


def _clean_json_response(raw_text: str) -> str:
    """Extract json from markdown fences or raw string."""
    cleaned = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()
    return cleaned


def parse_resume_with_gemini(resume_text: str) -> ParsedResume:
    """Parse resume text using Google Gemini API."""
    if not config.is_gemini_configured():
        raise ValueError("Google Gemini API key is not configured. Please set GEMINI_API_KEY in .env.")

    import google.generativeai as genai

    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name=config.DEFAULT_LLM_MODEL,
        system_instruction=SYSTEM_RESUME_PARSER_PROMPT,
        generation_config={"temperature": 0.1, "response_mime_type": "application/json"},
    )

    prompt = f"RESUME TEXT TO PARSE:\n{resume_text}"
    response = model.generate_content(prompt)
    raw_content = response.text

    cleaned_json = _clean_json_response(raw_content)
    parsed_dict = json.loads(cleaned_json)

    # Set raw text preview
    parsed_dict["raw_text_preview"] = resume_text[:500]
    parsed_dict["parsing_method"] = "gemini"

    return ParsedResume(**parsed_dict)


def parse_resume_heuristically(resume_text: str) -> ParsedResume:
    """
    Offline fallback parser using regex heuristics.
    Ensures the application remains testable and functional even when offline or without API credits.
    """
    lines = [line.strip() for line in resume_text.splitlines() if line.strip()]

    # Infer candidate name from first non-empty line
    name = "Candidate"
    if lines:
        first_line = lines[0]
        # Clean potential markers
        first_line = re.sub(r"^(Name[:\s\-]*)", "", first_line, flags=re.IGNORECASE).strip()
        if len(first_line.split()) <= 4 and not any(k in first_line.lower() for k in ["resume", "cv", "curriculum"]):
            name = first_line

    # Extract email
    email_match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", resume_text)
    email = email_match.group(0) if email_match else None

    # Extract phone
    phone_match = re.search(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,5}\)?[-.\s]?\d{3,5}[-.\s]?\d{3,5}", resume_text)
    phone = phone_match.group(0) if phone_match else None

    # Target Role
    target_role_match = re.search(r"(?:target\s*role|objective|desired\s*position)[:\s\-]+([^\n\r]+)", resume_text, re.IGNORECASE)
    if target_role_match:
        target_role = target_role_match.group(1).strip()
    else:
        # Check for common role keywords
        roles = ["Data Analyst", "Software Engineer", "Frontend Developer", "Backend Developer", "Machine Learning Engineer", "DevOps Engineer", "Data Scientist"]
        detected = [r for r in roles if re.search(rf"\b{re.escape(r)}\b", resume_text, re.IGNORECASE)]
        target_role = detected[0] if detected else "Software & Data Professional"

    # Extract Skills via common keyword dictionary
    common_skills = [
        "Python", "SQL", "Tableau", "Power BI", "Pandas", "NumPy", "Matplotlib", "Seaborn",
        "React", "TypeScript", "JavaScript", "HTML", "CSS", "Tailwind CSS", "Node.js",
        "FastAPI", "Flask", "Django", "PostgreSQL", "MySQL", "MongoDB", "Redis",
        "Docker", "Kubernetes", "AWS", "Git", "GitHub Actions", "CI/CD", "Linux",
        "PyTorch", "TensorFlow", "Scikit-Learn", "Machine Learning", "Deep Learning", "NLP",
        "Excel", "Data Cleaning", "EDA", "REST APIs", "Selenium", "Pytest"
    ]
    extracted_skills = []
    for s in common_skills:
        if re.search(rf"\b{re.escape(s)}\b", resume_text, re.IGNORECASE):
            extracted_skills.append(s)

    # Professional Summary
    summary_match = re.search(r"(?:summary|objective|profile)[:\s\-]+([\s\S]*?)(?=\n[A-Z\s]{4,}:|\n\n|\Z)", resume_text, re.IGNORECASE)
    summary = summary_match.group(1).strip() if summary_match else f"Aspiring {target_role} with proven technical foundations."

    # Experience heuristic extraction
    experience = []
    exp_sections = re.findall(r"(?:intern|engineer|analyst|developer|assistant|specialist)[^\n]+", resume_text, re.IGNORECASE)
    for exp_title in exp_sections[:3]:
        parts = exp_title.split("|")
        role_part = parts[0].strip()
        comp_part = parts[1].strip() if len(parts) > 1 else "Industry Organization"
        experience.append(ExperienceItem(
            role=role_part,
            company=comp_part,
            duration="Documented in Resume",
            bullets=["Executed assigned domain responsibilities as documented in resume text."]
        ))

    # Education heuristic extraction
    education = []
    edu_matches = re.findall(r"(?:bachelor|master|b\.tech|b\.e|b\.sc|m\.tech|degree|diploma)[^\n]+", resume_text, re.IGNORECASE)
    for edu in edu_matches[:2]:
        education.append(EducationItem(
            degree=edu.strip(),
            institution="Academic Institution",
            year="Completed",
            details=None
        ))

    return ParsedResume(
        name=name,
        email=email,
        phone=phone,
        location=None,
        target_role=target_role,
        summary=summary,
        skills=extracted_skills if extracted_skills else ["General Technical Skills"],
        experience=experience,
        education=education,
        raw_text_preview=resume_text[:500],
        parsing_method="offline_heuristic"
    )


def parse_resume(resume_text: str, prefer_offline: bool = False) -> ParsedResume:
    """
    Unified entry point for parsing resume text.
    Uses Google Gemini structured output if configured, otherwise falls back gracefully to offline heuristic.
    """
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is empty.")

    if not prefer_offline and config.is_gemini_configured():
        try:
            return parse_resume_with_gemini(resume_text)
        except Exception as e:
            # Fall back to heuristic on API failure, but note it in logs
            print(f"[SmartHireAI] Gemini parsing encountered: {e}. Falling back to offline heuristic parser.")
            parsed = parse_resume_heuristically(resume_text)
            parsed.parsing_method = "offline_heuristic (API fallback)"
            return parsed

    return parse_resume_heuristically(resume_text)
