import os
import re
import streamlit as st
from typing import Tuple, List

from pypdf import PdfReader

# ====== LangChain + OpenAI ======
from langchain_openai import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage

# ====== OPENAI Setup ======
os.environ["OPENAI_API_KEY"] = ""
llm = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo")

# ====== Utility Functions ======
def pdf_to_text(file_path: str) -> str:
    text_content = ""
    try:
        with open(file_path, "rb") as f:
            pdf = PdfReader(f)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text_content


def fetch_jd_template(role_type: str) -> str:
    templates = {
        "Software Engineer": """
        [Software Engineer JD Template]
        Title: Software Engineer
        Responsibilities:
        - Develop and maintain software applications.
        - Collaborate with cross-functional teams.
        - Troubleshoot and debug code.
        Requirements:
        - Proficient in Python and Java.
        - Experience with AWS.
        - 3+ years of experience in software development.
        Benefits:
        - Competitive salary.
        - Health insurance.
        - Remote work options.
        """,
        "Data Analyst": """
        [Data Analyst JD Template]
        Title: Data Analyst
        Responsibilities:
        - Analyze data to support business decisions.
        - Create data visualizations and reports.
        - Maintain databases and data systems.
        Requirements:
        - Proficient in SQL and Python.
        - Experience with data visualization tools like Tableau.
        - Strong analytical skills.
        Benefits:
        - Competitive salary.
        - Health insurance.
        - Professional development opportunities.
        """,
    }
    return templates.get(role_type, "No matching template found.")


def crosscheck_and_incorporate_requirements(draft_jd: str, requirements_list: List[str]) -> str:
    """
    Ensures each requirement is present. If missing, incorporate it under a 'Requirements' section.
    Returns an updated JD with no duplication blocks.
    """
    jd_lines = draft_jd.split("\n")

    # Find "Requirements:" heading (case-insensitive match).
    req_section_start = None
    for i, line in enumerate(jd_lines):
        if line.strip().lower().startswith("requirements:"):
            req_section_start = i
            break

    # If no Requirements heading found, append one.
    if req_section_start is None:
        jd_lines.append("Requirements:")
        req_section_start = len(jd_lines) - 1

    # Gather existing requirement bullets
    existing_req_bullets = []
    insertion_index = req_section_start + 1

    # Move insertion_index until the next heading or end
    while insertion_index < len(jd_lines):
        line_strip = jd_lines[insertion_index].strip()
        if line_strip and ":" in line_strip and line_strip.split(":", 1)[0].strip().lower() in [
            "responsibilities", "benefits", "title", "company overview",
            "additional info", "miscellaneous notes", "culture", "personality traits"
        ]:
            break
        # If it's a bullet under Requirements, store it
        if line_strip.startswith("-"):
            existing_req_bullets.append(line_strip[1:].strip())
        insertion_index += 1

    # For each requirement from the summary, check if it already exists in the draft
    requirements_to_add = []
    for req in requirements_list:
        if not any(req.lower() in bullet.lower() for bullet in existing_req_bullets):
            requirements_to_add.append(req)

    # Insert missing requirements as bullets
    for m_req in requirements_to_add:
        jd_lines.insert(insertion_index, f"- {m_req}")
        insertion_index += 1

    return "\n".join(jd_lines)


# ====== Agent Prompts ======

system_prompt = """
You are an advanced AI system specialized in generating professional, descriptive, 
and logically ordered Job Descriptions with the highest quality.

Final JD Structure (in order of importance):
1) Company Overview
2) Position/Title
3) Key Responsibilities
4) Qualifications/Requirements
5) Additional Info or Benefits
6) Culture Fit / Personality Traits
7) Application/Closing Statement

Rules:
- No \"###\" headings or repeated summary blocks.
- Use either plain text or simple bold headings (e.g., **Company Overview**).
- Provide well-organized, descriptive content with a professional tone.
- Incorporate any missing requirements into the final JD seamlessly.
- Avoid duplication: produce a single, clean JD.
"""

reader_prompt = """
You are Agent #1 (Reader/Analyzer).
Combine the inputs into a summary with these sections:

Role Type or Title: [...]
Responsibilities:
- [...]
Requirements:
- [...]
Miscellaneous Notes:
- [...]
If no input is provided for a section, use "Not specified" or "No requirements specified".
"""

template_selector_prompt = """
You are Agent #2 (Template Selector).
Pick the best matching JD template based on the role in the summary.
If none match, return "No matching template found."
"""

jd_drafter_prompt = """
You are Agent #3 (JD Drafter).
Create a cohesive draft JD using:
- The summary from Agent #1
- The chosen template from Agent #2
Follow the final structure from the system prompt, but do not replicate the summary text verbatim.
Use descriptive language and maintain a professional style.
"""

crosscheck_prompt = """
You are Agent #4 (Crosschecker).
Check if the 'Requirements' from the summary are fully present in the JD. 
If any are missing, list them. Else say 'All found.'
"""

qa_specialist_prompt = """
You are Agent #5 (QA Specialist).
Polish grammar, remove redundancies, ensure the headings match the final structure:
1) Company Overview
2) Position/Title
3) Key Responsibilities
4) Qualifications/Requirements
5) Additional Info or Benefits
6) Culture Fit / Personality Traits
7) Application/Closing Statement

- Use bold headings (like **Company Overview**).
- No triple-hash headings or repeated blocks.
- Return the final refined JD.
"""


# ====== JD Generator Pipeline ======
def validate_inputs(hr_input: str, manager_input: str, pdf_text: str) -> bool:
    return bool(hr_input.strip() or manager_input.strip() or pdf_text.strip())

def run_jd_generator(hr_input: str = "", manager_input: str = "", pdf_path: str = None) -> str:
    # Optional PDF text
    pdf_text = ""
    if pdf_path:
        try:
            pdf_text = pdf_to_text(pdf_path)
        except Exception as e:
            return f"Error reading PDF: {e}"

    # Validate
    if not validate_inputs(hr_input, manager_input, pdf_text):
        return "Error: Please provide some information."

    # 1) Agent #1: Reader/Analyzer
    agent1_messages = [
        SystemMessage(content=system_prompt + reader_prompt),
        HumanMessage(content=f"""
Inputs:
HR Input: {hr_input if hr_input.strip() else 'None'}
Manager Input: {manager_input if manager_input.strip() else 'None'}
PDF Text: {pdf_text if pdf_text.strip() else 'None'}
""")
    ]
    agent1_response = llm(agent1_messages)
    summary_of_role = agent1_response.content

    # Extract bullet requirements
    req_pattern = re.compile(r"Requirements:\s*(?:-\s*[^\n]+)+", re.IGNORECASE)
    match = req_pattern.search(summary_of_role)
    requirements = []
    if match:
        bullet_pattern = re.compile(r"-\s*([^\n]+)")
        found_bullets = bullet_pattern.findall(match.group(0))
        for item in found_bullets:
            item_str = item.strip()
            if item_str.lower() not in ["no requirements specified", ""]:
                requirements.append(item_str)

    # 2) Agent #2: Template Selector
    agent2_messages = [
        SystemMessage(content=system_prompt + template_selector_prompt),
        HumanMessage(content=f"Job Summary:\n{summary_of_role}")
    ]
    agent2_response = llm(agent2_messages)
    chosen_template = agent2_response.content.strip()
    if chosen_template == "No matching template found.":
        return "Error: No matching template found."

    # 3) Agent #3: JD Drafter
    agent3_messages = [
        SystemMessage(content=system_prompt + jd_drafter_prompt),
        HumanMessage(content=f"SUMMARY:\n{summary_of_role}\n\nTEMPLATE:\n{chosen_template}")
    ]
    draft_jd_response = llm(agent3_messages)
    draft_jd = draft_jd_response.content

    # 4) Agent #4: Crosschecker
    agent4_messages = [
        SystemMessage(content=system_prompt + crosscheck_prompt),
        HumanMessage(content=f"Draft JD:\n{draft_jd}\n\nSUMMARY:\n{summary_of_role}")
    ]
    crosscheck_resp = llm(agent4_messages).content.strip()
    # Check if any are missing
    missing_pattern = re.compile(r"-\s*(.+)")
    missing_list = missing_pattern.findall(crosscheck_resp) if "found" not in crosscheck_resp.lower() else []

    # Incorporate missing
    final_jd_incorporated = crosscheck_and_incorporate_requirements(draft_jd, requirements)

    # 5) Agent #5: QA Specialist
    agent5_messages = [
        SystemMessage(content=system_prompt + qa_specialist_prompt),
        HumanMessage(content=f"""
Here is the final JD (with missing requirements incorporated if any):

{final_jd_incorporated}
""")
    ]
    agent5_response = llm(agent5_messages)
    polished_jd = agent5_response.content.strip()

    return polished_jd


# ====== Streamlit App with 4 Page Flow ======
def main():
    st.title("Multi-Page JD Generator")

    if "page" not in st.session_state:
        st.session_state.page = 1

    # Page 1
    if st.session_state.page == 1:
        st.header("Page 1 - Company Culture & Values")
        company_name = st.text_input("Company Name")
        culture_fit_factors = st.text_input("Culture Fit Factors (e.g., integrity, leadership)")
        personality_traits = st.text_input("Personality Traits")

        if st.button("Next Page"):
            st.session_state.page1_data = {
                "company_name": company_name,
                "culture_fit_factors": culture_fit_factors,
                "personality_traits": personality_traits
            }
            st.session_state.page = 2

    # Page 2
    elif st.session_state.page == 2:
        st.header("Page 2 - Manager's Requirements & Insights")
        work_experience = st.text_area(
            "Work Experience Required",
            placeholder="3+ years with Python, Docker, AWS, etc."
        )
        technology_domains_tools = st.text_area(
            "Technologies, Domains, Tools",
            placeholder="Flask, Kubernetes, Streaming systems, etc."
        )

        if st.button("Next Page"):
            st.session_state.page2_data = {
                "work_experience": work_experience,
                "technology_domains_tools": technology_domains_tools
            }
            st.session_state.page = 3

    # Page 3
    elif st.session_state.page == 3:
        st.header("Page 3 - Industry Standards & Customizations")
        role_title = st.text_input("Role Title", placeholder="Senior Software Engineer")
        core_responsibilities = st.text_area("Core Responsibilities",
                                             placeholder="- Lead backend architecture\n- Collaborate with DevOps...")
        skills_competence = st.text_area("Skills & Competence",
                                         placeholder="- Strong debugging\n- Excellent communication...")
        education_requirements = st.text_input("Education Requirements", placeholder="Bachelor's in CS or related field")

        if st.button("Next Page"):
            st.session_state.page3_data = {
                "role_title": role_title,
                "core_responsibilities": core_responsibilities,
                "skills_competence": skills_competence,
                "education_requirements": education_requirements
            }
            st.session_state.page = 4

    # Page 4
    elif st.session_state.page == 4:
        st.header("Page 4 - Job-Specific Customization")
        specialized_role_focus = st.text_area("Specialized Role Focus",
                                              placeholder="Focus on microservices, real-time analytics, etc.")
        unique_skills = st.text_area("Unique Skills (Optional)",
                                     placeholder="E.g., Machine learning frameworks, etc.")
        key_deliverables = st.text_area("Key Deliverables (Optional)",
                                        placeholder="Monthly releases, code reviews, mentorship, etc.")

        if st.button("Finalize JD"):
            st.session_state.page4_data = {
                "specialized_role_focus": specialized_role_focus,
                "unique_skills": unique_skills,
                "key_deliverables": key_deliverables
            }
            st.session_state.page = 5

    # Final Page
    elif st.session_state.page == 5:
        st.header("JD Generator - Final Result")

        page1_data = st.session_state.get("page1_data", {})
        page2_data = st.session_state.get("page2_data", {})
        page3_data = st.session_state.get("page3_data", {})
        page4_data = st.session_state.get("page4_data", {})

        # Combine into HR Input
        hr_input = f"""
Company Name: {page1_data.get("company_name", "")}
Culture Fit Factors: {page1_data.get("culture_fit_factors", "")}
Personality Traits: {page1_data.get("personality_traits", "")}

Work Experience Required:
{page2_data.get("work_experience", "")}

Technologies, Domains, Tools:
{page2_data.get("technology_domains_tools", "")}

Role Title: {page3_data.get("role_title", "")}
Core Responsibilities:
{page3_data.get("core_responsibilities", "")}

Skills & Competence:
{page3_data.get("skills_competence", "")}

Education Requirements:
{page3_data.get("education_requirements", "")}

Specialized Role Focus:
{page4_data.get("specialized_role_focus", "")}

Unique Skills:
{page4_data.get("unique_skills", "")}

Key Deliverables:
{page4_data.get("key_deliverables", "")}
""".strip()

        manager_input = ""

        jd_text = run_jd_generator(hr_input=hr_input, manager_input=manager_input)

        # Display the final JD with visible text
        html_content = jd_text.replace('\n', '<br>')
        st.markdown(
            f"<div style='background-color: #f9f9f9; color: black; padding: 1em; border-radius: 5px;'>"
            f"{html_content}"
            f"</div>",
            unsafe_allow_html=True
        )

        st.write("—" * 40)
        if st.button("Start Over"):
            st.session_state.clear()
            st.write("Session cleared. Reload or go to Page 1.")
            st.stop()


if __name__ == "__main__":
    main()
