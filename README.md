# Resume Screening System Using NLP

An AI-powered Resume Screening System that analyzes resumes against job descriptions and calculates an ATS (Applicant Tracking System) score using Natural Language Processing techniques.

This project helps candidates understand how their resume matches job requirements and suggests improvements.

---

## 🚀 Features

- Upload resume in **PDF format**
- Paste a **job description**
- Automatic **job role detection**
- Resume similarity using **TF-IDF + Cosine Similarity**
- **Matched skills detection**
- **Missing skills identification**
- **ATS score calculation**
- **Skill gap analysis**
- **Resume improvement suggestions**
- **Interactive Streamlit dashboard**

---

## 🧠 How It Works

1. Resume text is extracted from the uploaded PDF using **pdfplumber**
2. Text is cleaned using **NLP preprocessing**
3. Job description and resume are converted into vectors using **TF-IDF**
4. **Cosine similarity** calculates resume-job matching score
5. Skills are extracted from job description and compared with resume
6. The system generates:

- Resume Match Score
- Skill Match Percentage
- Final ATS Score
- Missing Skills
- Resume Improvement Suggestions

---

## 🛠 Tech Stack

- **Python**
- **Natural Language Processing (NLP)**
- **Scikit-learn**
- **Streamlit**
- **pdfplumber**
- **NLTK**
- **Matplotlib**

---

## 📊 Project Output

The system provides:

- Resume Match Score
- Final ATS Score
- Matched Skills
- Missing Skills
- Skill Gap Analysis
- Resume Improvement Suggestions

---

## ▶️ How to Run the Project

### 1️⃣ Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Resume-Screening-System.git

### 2️⃣ Navigate to the folder
cd AI-Resume-Screening-System

### 3️⃣ Install dependencies
pip install -r requirements(resume).txt

### 4️⃣ Run the Streamlit app
streamlit run app.py
