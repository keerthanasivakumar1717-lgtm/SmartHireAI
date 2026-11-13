"""
SmartHireAI - Guardrails and Responsible AI Safety Layer
Validates user inputs, enforces career advisory scope, prevents prompt injections,
and sanitizes untrusted input before LLM invocations.
"""

from typing import Tuple, Optional
import re

# Career-relevant domain keywords
CAREER_KEYWORDS = {
    "career", "job", "resume", "cv", "interview", "skills", "salary", "hiring", "recruiter",
    "internship", "portfolio", "project", "education", "experience", "promotion", "switch",
    "transition", "coding", "software", "analyst", "engineer", "developer", "sql", "python",
    "data", "tableau", "aws", "cloud", "docker", "frontend", "backend", "fullstack", "ml",
    "machine learning", "devops", "qa", "testing", "degree", "certification", "star method",
    "cover letter", "linkedin", "ats", "work", "role", "position", "guidance", "roadmap",
    "learn", "course", "bootcamp", "technical", "behavioral", "weakness", "strength"
}

# Prompt injection and jailbreak patterns
PROMPT_INJECTION_PATTERNS = [
    r"ignore (?:all )?(?:previous|above|system)?\s*(?:instructions|rules|prompts)",
    r"disregard (?:all )?(?:previous|above|system)?\s*(?:instructions|prompts)",
    r"(?:what is|show me|print|reveal|output|display|dump) (?:your|the)?\s*(?:system prompt|instructions|secret|api key|keys)",
    r"you are now (?:dan|an unfiltered|a new|jailbroken)",
    r"bypass (?:safety|guardrails|filters)",
    r"drop database|exec xp_cmdshell|<script>|union select",
    r"system override",
    r"base64 decode",
]

# Explicit out-of-scope topics
OUT_OF_SCOPE_PATTERNS = [
    (r"\b(?:diagnose|medical|prescription|symptom|illness|disease|doctor|medication)\b",
     "I am an AI Career Mentor designed specifically to assist with career advice, resumes, and interview preparation. I cannot provide medical or health-related advice."),
    (r"\b(?:crypto investment|stock tip|financial trading|lottery|forex)\b",
     "I specialize in career navigation and job readiness, and cannot provide financial or investment advice."),
    (r"\b(?:recipe|cooking|bake a cake|ingredients for soup)\b",
     "My expertise is dedicated to career mentoring, resumes, and job matching. Please ask a career- or tech-related question!"),
    (r"\b(?:write a song|write a poem about love|fiction story)\b",
     "I am tuned specifically for professional career development, technical roadmaps, and resume improvements."),
]


def sanitize_input(text: str) -> str:
    """
    Sanitize untrusted text to neutralize potential delimiter collision or control characters.
    """
    if not text:
        return ""
    # Strip null and terminal escape sequences
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Neutralize markdown system delimiter spoofs
    sanitized = sanitized.replace("```system", "'''system").replace("```prompt", "'''prompt")
    return sanitized.strip()


def validate_user_query(query: str) -> Tuple[bool, Optional[str]]:
    """
    Validate incoming user query against safety rules, prompt injection, and career scope.
    Returns: (is_valid: bool, refusal_reason: Optional[str])
    """
    if not query or not query.strip():
        return False, "Query is empty. Please enter a career-related question."

    clean_query = sanitize_input(query)

    # 1. Check for excessive length (DoS / context blowup prevention)
    if len(clean_query) > 2000:
        return False, "Your question is too long. Please condense your query to under 2,000 characters."

    query_lower = clean_query.lower()

    # 2. Check for Prompt Injection / System Tampering
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, query_lower):
            return False, (
                "Request declined. I cannot alter my core operational parameters, "
                "reveal system instructions, or bypass safety guardrails. "
                "How can I help with your career questions today?"
            )

    # 3. Check for Out-of-Scope Domains
    for pattern, refusal in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, query_lower):
            return False, refusal

    # 4. Check for Career Scope Relevance (for non-trivial queries)
    words = set(re.findall(r"\b[a-zA-Z]{3,}\b", query_lower))
    # Allow common polite greetings
    greetings = {"hello", "hi", "hey", "greetings", "good", "morning", "afternoon", "evening", "thanks", "thank", "help"}
    if words.issubset(greetings):
        return True, None

    # Check keyword overlap
    overlap = words.intersection(CAREER_KEYWORDS)
    if not overlap and len(words) > 6:
        # If a substantive question contains zero career keywords, request refocus
        return False, (
            "I am SmartHire's Career Mentor. My focus is strictly on career planning, "
            "technical role roadmaps, interview preparation, and resume alignment. "
            "Please ask a question related to these topics."
        )

    return True, None
