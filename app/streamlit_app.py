"""
SmartHireAI - Streamlit Career Intelligence Portal
Full multi-step web application integrating Resume Parsing, Semantic Job Matching (FAISS),
CV Improvement Generation, and Grounded Career Mentor (RAG).
"""

import sys
from pathlib import Path

# Add project root to sys.path to enable direct execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from src import config
from src.parsing.loader import load_resume_document, DocumentLoaderError
from src.parsing.resume_parser import parse_resume, ParsedResume
from src.search.job_search import get_default_job_searcher, JobSearchIndex
from src.generate.cv_suggestions import generate_cv_improvements
from src.mentor.rag_chain import get_career_mentor

# Page Configuration
st.set_page_config(
    page_title="SmartHireAI — Career & Resume Intelligence Portal",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #64748B; margin-bottom: 1.5rem; }
    .metric-card { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; margin-bottom: 12px; }
    .skill-badge { display: inline-block; background-color: #EEF2FF; color: #4338CA; padding: 4px 10px; border-radius: 12px; font-size: 0.82rem; margin: 2px 4px 2px 0; font-weight: 500; }
    .skill-badge-matched { display: inline-block; background-color: #ECFDF5; color: #065F46; padding: 4px 10px; border-radius: 12px; font-size: 0.82rem; margin: 2px 4px 2px 0; font-weight: 500; }
    .skill-badge-missing { display: inline-block; background-color: #FEF3C7; color: #92400E; padding: 4px 10px; border-radius: 12px; font-size: 0.82rem; margin: 2px 4px 2px 0; font-weight: 500; }
    .score-badge { font-weight: 700; font-size: 1.1rem; padding: 4px 12px; border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "parsed_resume" not in st.session_state:
    st.session_state.parsed_resume = None
if "matched_jobs" not in st.session_state:
    st.session_state.matched_jobs = []
if "selected_job" not in st.session_state:
    st.session_state.selected_job = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "cv_report" not in st.session_state:
    st.session_state.cv_report = ""

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/briefcase.png", width=64)
    st.title("SmartHireAI")
    st.caption("GenAI Resume Matching & AI Career Mentor")

    st.markdown("---")
    st.subheader("System Status")
    is_gemini = config.is_gemini_configured()
    if is_gemini:
        st.success(f"🟢 Gemini API: Active ({config.DEFAULT_LLM_MODEL})")
    else:
        st.warning("🟡 Gemini API: Not Configured (Using Offline Heuristics)")
        st.caption("Add `GEMINI_API_KEY` to `.env` for cloud LLM reasoning.")

    st.info(f"🔍 Vector Engine: FAISS / {config.DEFAULT_EMBEDDING_MODEL.split('/')[-1]}")

    st.markdown("---")
    st.subheader("Data & Privacy")
    st.caption("🔒 **Ephemeral Processing:** Uploaded resumes are held only in memory during your active session and are never written to permanent cloud storage.")

    st.markdown("---")
    if st.button("🔄 Rebuild FAISS Job Index", use_container_width=True):
        with st.spinner("Rebuilding FAISS index from dataset..."):
            try:
                searcher = JobSearchIndex()
                count = searcher.build_index()
                st.success(f"Index built successfully ({count} jobs)!")
            except Exception as e:
                st.error(f"Failed to build index: {e}")

# Header
st.markdown('<div class="main-header">SmartHireAI Career Intelligence Portal</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">AI-Powered Resume Structuring, Semantic Job Matching, CV Optimization, and Grounded Career Guidance</div>',
    unsafe_allow_html=True,
)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📄 1. Resume Upload & Profile",
    "🎯 2. Matched Jobs",
    "💡 3. CV Improvement",
    "🤖 4. AI Career Mentor (RAG)"
])

# ==============================================================================
# TAB 1: RESUME UPLOAD & PROFILE REVIEW
# ==============================================================================
with tab1:
    st.header("Step 1: Upload & Parse Your Resume")
    col1, col2 = st.columns([3, 2])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload your CV (PDF, Word DOCX, or TXT)",
            type=["pdf", "docx", "txt"],
            help="Files are processed locally and securely in-memory.",
        )

    with col2:
        st.write("Or test with pre-loaded sample profiles:")
        sample_choice = st.selectbox(
            "Choose a demo resume:",
            ["Select sample...", "Data Analyst (Aakash Sharma)", "Backend SWE (Priya Verma)", "Frontend Developer (Rohan Mehta)"]
        )
        if st.button("Load Selected Sample Resume", use_container_width=True):
            sample_map = {
                "Data Analyst (Aakash Sharma)": "sample_resume_data_analyst.txt",
                "Backend SWE (Priya Verma)": "sample_resume_swe.txt",
                "Frontend Developer (Rohan Mehta)": "sample_resume_frontend.txt"
            }
            if sample_choice in sample_map:
                sample_path = config.RESUMES_DATA_DIR / sample_map[sample_choice]
                if sample_path.exists():
                    with open(sample_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    with st.spinner("Parsing sample resume profile..."):
                        profile = parse_resume(text)
                        st.session_state.parsed_resume = profile
                        # Auto match jobs
                        searcher = get_default_job_searcher()
                        st.session_state.matched_jobs = searcher.search(
                            profile.to_profile_summary_text(),
                            candidate_skills=profile.skills,
                            top_k=config.DEFAULT_TOP_K
                        )
                        st.session_state.selected_job = st.session_state.matched_jobs[0] if st.session_state.matched_jobs else None
                    st.success(f"Loaded profile: {profile.name} ({profile.target_role})")

    # If file uploaded
    if uploaded_file is not None:
        if st.button("Parse Uploaded Resume", type="primary"):
            with st.spinner("Extracting text and extracting structured candidate profile..."):
                try:
                    resume_text = load_resume_document(uploaded_file, filename_hint=uploaded_file.name)
                    profile = parse_resume(resume_text)
                    st.session_state.parsed_resume = profile

                    # Search matching jobs immediately
                    searcher = get_default_job_searcher()
                    st.session_state.matched_jobs = searcher.search(
                        profile.to_profile_summary_text(),
                        candidate_skills=profile.skills,
                        top_k=config.DEFAULT_TOP_K
                    )
                    st.session_state.selected_job = st.session_state.matched_jobs[0] if st.session_state.matched_jobs else None
                    st.success(f"Successfully parsed resume for {profile.name}!")
                except DocumentLoaderError as de:
                    st.error(f"Document Loading Error: {de}")
                except Exception as e:
                    st.error(f"Error parsing resume: {e}")

    # Display Parsed Profile
    if st.session_state.parsed_resume:
        prof: ParsedResume = st.session_state.parsed_resume
        st.markdown("---")
        st.subheader(f"Candidate Profile: {prof.name}")
        st.caption(f"Parser Engine: `{prof.parsing_method}`")

        c1, c2, c3 = st.columns([1, 1, 1])
        c1.metric("Target Specialization", prof.target_role)
        c2.metric("Identified Skills", len(prof.skills))
        c3.metric("Experience Entries", len(prof.experience))

        st.markdown("#### Professional Summary")
        st.info(prof.summary or "No summary provided in resume.")

        st.markdown("#### Identified Technical & Domain Skills")
        skills_html = "".join([f'<span class="skill-badge">{s}</span>' for s in prof.skills])
        st.markdown(skills_html or "*No specific skills identified.*", unsafe_allow_html=True)

        col_exp, col_edu = st.columns(2)
        with col_exp:
            st.markdown("#### Experience Highlights")
            if prof.experience:
                for exp in prof.experience:
                    st.markdown(f"**{exp.role}** — *{exp.company or 'Organization'}* ({exp.duration or 'N/A'})")
                    for b in exp.bullets:
                        st.markdown(f"- {b}")
            else:
                st.caption("No experience entries recorded.")

        with col_edu:
            st.markdown("#### Academic Qualifications")
            if prof.education:
                for edu in prof.education:
                    st.markdown(f"**{edu.degree}**")
                    st.caption(f"{edu.institution or 'University'} | Year: {edu.year or 'N/A'}")
                    if edu.details:
                        st.caption(f"Details: {edu.details}")
            else:
                st.caption("No education entries recorded.")

        with st.expander("🔍 View Raw Extracted Resume Text Preview"):
            st.text(prof.raw_text_preview)

        st.markdown("👉 **Next Step:** Switch to the **'🎯 2. Matched Jobs'** tab to view your semantic recommendations.")
    else:
        st.info("👆 Please upload a resume or click one of the demo buttons above to begin.")

# ==============================================================================
# TAB 2: SEMANTIC JOB MATCHES
# ==============================================================================
with tab2:
    st.header("Step 2: Semantic Job Recommendations")

    if not st.session_state.parsed_resume:
        st.warning("Please upload or load a resume in Tab 1 first.")
    else:
        prof = st.session_state.parsed_resume
        st.write(f"Displaying jobs semantically matched for **{prof.name}** based on target role: *{prof.target_role}* and {len(prof.skills)} skills.")

        top_k_select = st.slider("Number of jobs to recommend (Top-K):", min_value=1, max_value=10, value=config.DEFAULT_TOP_K)

        if st.button("Refresh Matches", key="refresh_matches"):
            searcher = get_default_job_searcher()
            st.session_state.matched_jobs = searcher.search(
                prof.to_profile_summary_text(),
                candidate_skills=prof.skills,
                top_k=top_k_select
            )

        matches = st.session_state.matched_jobs
        if not matches:
            st.info("No matching jobs found. Click 'Refresh Matches' or verify the job dataset.")
        else:
            for idx, job in enumerate(matches, 1):
                score = job.get("match_percentage", 0.0)
                color = "#059669" if score >= 75 else ("#D97706" if score >= 60 else "#4B5563")

                with st.container():
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div>
                                <h3 style="margin:0; color:#0F172A;">{idx}. {job['title']}</h3>
                                <p style="margin:2px 0 8px 0; color:#475569; font-weight:500;">
                                    🏢 {job['company']} &nbsp;|&nbsp; 📍 {job.get('location', 'Location N/A')} &nbsp;|&nbsp; 🎓 {job.get('experience_level', 'Experience N/A')}
                                </p>
                            </div>
                            <div style="text-align:right;">
                                <span class="score-badge" style="background-color:{color}20; color:{color}; border:1px solid {color};">
                                    {score}% Match
                                </span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Skills Breakdown
                    c_m, c_miss = st.columns(2)
                    with c_m:
                        st.markdown("**Matched Skills on Your Resume:**")
                        matched_html = "".join([f'<span class="skill-badge-matched">✓ {s}</span>' for s in job.get("matched_skills", [])])
                        st.markdown(matched_html or "*None directly matched.*", unsafe_allow_html=True)

                    with c_miss:
                        st.markdown("**Missing Skills to Bridge:**")
                        missing_html = "".join([f'<span class="skill-badge-missing">+ {s}</span>' for s in job.get("missing_skills", [])])
                        st.markdown(missing_html or "*No gaps identified!*", unsafe_allow_html=True)

                    # Expand description & Selection button
                    col_b1, col_b2 = st.columns([3, 1])
                    with col_b1:
                        with st.expander(f"Read Full Job Description for {job['title']}"):
                            st.write(job.get("job_description", "No description available."))
                    with col_b2:
                        if st.button(f"Analyze for CV Tips", key=f"select_job_{idx}", use_container_width=True):
                            st.session_state.selected_job = job
                            st.session_state.cv_report = ""  # Trigger re-generation
                            st.success(f"Selected '{job['title']}'! Now switch to Tab 3 for feedback.")

# ==============================================================================
# TAB 3: CV IMPROVEMENT GENERATOR
# ==============================================================================
with tab3:
    st.header("Step 3: CV Optimization & Skill Gap Feedback")

    if not st.session_state.parsed_resume:
        st.warning("Please upload a resume in Tab 1 first.")
    elif not st.session_state.selected_job:
        st.info("Please select a target job from Tab 2 (click 'Analyze for CV Tips' on any job card).")
    else:
        target_job = st.session_state.selected_job
        prof = st.session_state.parsed_resume

        st.markdown(f"### Target Role: **{target_job['title']}** at **{target_job['company']}**")

        if not st.session_state.cv_report:
            if st.button("Generate Detailed CV Improvement Report", type="primary"):
                with st.spinner("Analyzing skill gaps, reviewing bullet points, and tailoring summary..."):
                    report = generate_cv_improvements(prof, target_job)
                    st.session_state.cv_report = report

        if st.session_state.cv_report:
            st.markdown(st.session_state.cv_report)
            if st.button("Regenerate Improvement Report", key="regen_cv"):
                with st.spinner("Re-analyzing with fresh recommendations..."):
                    report = generate_cv_improvements(prof, target_job)
                    st.session_state.cv_report = report
                    st.rerun()

# ==============================================================================
# TAB 4: AI CAREER MENTOR (RAG)
# ==============================================================================
with tab4:
    st.header("Step 4: AI Career Mentor")
    st.caption("Ask questions about career switching, technical skills, interviews, or role expectations. Answers are strictly grounded in our career guides and job corpus.")

    mentor = get_career_mentor()

    # Pre-set starter questions
    st.write("**Quick Inspiration Questions:**")
    q_cols = st.columns(3)
    preset_q = None
    if q_cols[0].button("How do I transition to Data Analyst?"):
        preset_q = "How do I transition to Data Analyst?"
    if q_cols[1].button("How should I use the STAR method in interviews?"):
        preset_q = "How should I use the STAR method in interviews?"
    if q_cols[2].button("What makes a strong Backend Engineer resume?"):
        preset_q = "What makes a strong Backend Engineer resume?"

    # Display Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources Consulted"):
                    for s in message["sources"]:
                        st.markdown(f"- **{s['name']}** (Section: *{s['section']}*, Match: {s.get('relevance_score', 'N/A')}%)")

    # Handle Input
    user_query = st.chat_input("Ask a career question (e.g. 'What projects differentiate an entry-level analyst?')...")
    active_prompt = preset_q or user_query

    if active_prompt:
        # Append User Message
        st.session_state.chat_history.append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        # Generate Grounded Response
        with st.chat_message("assistant"):
            with st.spinner("Searching career knowledge base and formulating grounded response..."):
                response_dict = mentor.answer_question(
                    active_prompt,
                    chat_history=st.session_state.chat_history[:-1]
                )
                answer = response_dict["answer"]
                sources = response_dict.get("sources", [])

                st.markdown(answer)
                if sources:
                    with st.expander("📚 Sources Consulted"):
                        for s in sources:
                            st.markdown(f"- **{s['name']}** (Section: *{s['section']}*, Match: {s.get('relevance_score', 'N/A')}%)")

                # Append Assistant Message
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
