import streamlit as st
import pdfplumber
import re
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords")

# -----------------------------
# Extract text from PDF
# -----------------------------
def extract_text_from_pdf(pdf_file):

    text = ""

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()

    return text


# -----------------------------
# Text preprocessing
# -----------------------------
stop_words = set(stopwords.words("english"))

def preprocess(text):

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = text.lower()

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)


# -----------------------------
# Similarity score
# -----------------------------
def calculate_similarity(resume_text, job_text):

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform([resume_text, job_text])

    similarity = cosine_similarity(vectors[0:1], vectors[1:2])

    return round(similarity[0][0] * 100, 2)


# -----------------------------
# Skills database
# -----------------------------
skills_db = {

"programming":[
"python","r","sql","scala","java","c++"
],

"data_analysis":[
"pandas","numpy","data analysis","data cleaning","data preprocessing",
"exploratory data analysis","eda","data visualization","feature engineering"
],

"machine_learning":[
"machine learning","supervised learning","unsupervised learning",
"regression","classification","clustering","decision trees",
"random forest","gradient boosting","xgboost"
],

"deep_learning":[
"deep learning","neural networks","cnn","rnn","lstm",
"tensorflow","keras","pytorch"
],

"nlp":[
"natural language processing","nlp","text classification",
"sentiment analysis","bert"
],

"data_visualization":[
"matplotlib","seaborn","plotly","power bi","tableau","dashboard"
],

"statistics_math":[
"statistics","probability","hypothesis testing"
],

"big_data":[
"hadoop","spark","pyspark","airflow"
],

"databases":[
"mysql","postgresql","mongodb"
],

"cloud":[
"aws","azure","gcp"
],

"tools":[
"git","github","excel","jupyter"
]

}


# -----------------------------
# Job roles
# -----------------------------
job_roles = {

"Data Scientist":[
"data scientist","machine learning","predictive modeling",
"statistical analysis","feature engineering"
],

"Data Analyst":[
"data analyst","data analysis","dashboard","power bi",
"tableau","business intelligence"
],

"Machine Learning Engineer":[
"machine learning engineer","model deployment","mlops"
],

"AI Engineer":[
"ai engineer","deep learning","neural networks","nlp"
],

"Data Engineer":[
"data engineer","data pipelines","etl","spark","airflow"
]

}


# -----------------------------
# Regex skill detection
# -----------------------------
def skill_exists(skill, text):

    pattern = r"\b" + re.escape(skill.lower()) + r"\b"

    return re.search(pattern, text.lower()) is not None


# -----------------------------
# Matched skills
# -----------------------------
def get_matched_skills(resume_text, job_text):

    matched = set()

    for category in skills_db.values():
        for skill in category:

            if skill_exists(skill, job_text) and skill_exists(skill, resume_text):
                matched.add(skill)

    return sorted(list(matched))


# -----------------------------
# Missing skills
# -----------------------------
def get_missing_skills(resume_text, job_text):

    missing = set()

    for category in skills_db.values():
        for skill in category:

            if skill_exists(skill, job_text) and not skill_exists(skill, resume_text):
                missing.add(skill)

    return sorted(list(missing))


# -----------------------------
# Detect job role
# -----------------------------
def detect_job_role(job_text):

    roles = []

    for role, keywords in job_roles.items():
        for keyword in keywords:
            if keyword.lower() in job_text.lower():
                roles.append(role)
                break

    return roles


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("AI Resume Screening System")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

job_description = st.text_area("Paste Job Description")


# -----------------------------
# Analyze button
# -----------------------------
if st.button("Analyze Resume"):

    if uploaded_file is not None and job_description != "":

        resume_text = extract_text_from_pdf(uploaded_file)

        resume_clean = preprocess(resume_text)

        job_clean = preprocess(job_description)

        # Detect job role
        roles = detect_job_role(job_description)

        st.subheader("Detected Job Role")

        if len(roles) == 0:
            st.write("General Data Role")
        else:
            for role in roles:
                st.write("🎯", role)

        st.divider()

        # Resume similarity
        score = calculate_similarity(resume_clean, job_clean)

        st.subheader(f"Resume Match Score: {score}%")

        st.divider()

        # Skills detection
        matched = get_matched_skills(resume_text, job_description)
        missing = get_missing_skills(resume_text, job_description)

        total_skills = len(matched) + len(missing)

        if total_skills > 0:
            skill_match_percentage = (len(matched) / total_skills) * 100
        else:
            skill_match_percentage = 0

        final_ats_score = (score * 0.6) + (skill_match_percentage * 0.4)

        st.subheader(f"Final ATS Score: {round(final_ats_score,2)}%")

        st.progress(final_ats_score / 100)

        # Resume strength
        if final_ats_score < 40:
            st.error("⚠️ Resume Strength: Weak – Improve skills")

        elif final_ats_score < 70:
            st.warning("👍 Resume Strength: Moderate – Some improvements needed")

        else:
            st.success("🚀 Resume Strength: Strong – Good match")

        st.divider()

        # Skill match
        st.subheader(f"Skill Match: {round(skill_match_percentage,2)}%")

        st.progress(skill_match_percentage / 100)

        st.divider()

        # Skill summary
        st.subheader("Skill Summary")

        col1, col2 = st.columns(2)

        col1.metric("Matched Skills", len(matched))
        col2.metric("Missing Skills", len(missing))

        st.divider()

        # Matched skills
        st.subheader("Matched Skills")

        if len(matched) == 0:
            st.write("No matching skills found.")
        else:
            for skill in matched:
                st.write("✅", skill)

        st.divider()

        # Missing skills
        st.subheader("Missing Skills")

        if len(missing) == 0:
            st.write("Your resume matches most required skills.")
        else:
            for skill in missing:
                st.write("❌", skill)

        st.divider()

        # Visualization
        st.subheader("Skill Gap Analysis")

        labels = ["Matched Skills", "Missing Skills"]
        values = [len(matched), len(missing)]

        fig, ax = plt.subplots()

        ax.bar(labels, values)

        ax.set_ylabel("Number of Skills")

        ax.set_title("Skill Gap Visualization")

        st.pyplot(fig)

        st.divider()

        # AI suggestions
        st.subheader("AI Resume Improvement Suggestions")

        if len(missing) == 0:
            st.success("Your resume already contains most required skills.")
        else:
            st.write("To improve your ATS score, consider adding:")

            for skill in missing[:5]:
                st.write("📌", skill)
                
        st.divider()

        st.subheader("Why Your ATS Score Is This Value")

        # Case 1: Skills high but similarity low
        if skill_match_percentage >= 80 and score < 50:

            st.warning(
                "Your skills match the job requirements, but the resume content is not closely aligned with the job description."
            )

            st.write("Possible reasons:")
            st.write("• Your resume may not contain enough keywords from the job description")
            st.write("• Your project descriptions may use different wording than the job posting")
            st.write("• Some job responsibilities are not clearly mentioned in your resume")

            st.subheader("How to Improve")

            st.write("• Use similar keywords from the job description in your resume")
            st.write("• Rewrite project descriptions using job-specific terms")
            st.write("• Highlight relevant experience related to this role")


        # Case 2: Skills missing
        elif len(missing) > 0 and skill_match_percentage < 70:

            st.warning("Your ATS score is low mainly because some required skills are missing.")

            st.write("Consider adding these skills to improve your resume:")

            for skill in missing[:5]:
                st.write("📌", skill)


        # Case 3: Resume moderately aligned
        elif score >= 50 and score < 70:

            st.info(
                "Your resume partially matches the job description. Improving keywords and highlighting relevant experience can increase your ATS score."
            )


        # Case 4: Strong resume
        else:

            st.success(
                "Your resume is well aligned with the job description and should perform well in ATS screening."
            )