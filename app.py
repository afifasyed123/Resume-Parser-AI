import streamlit as st
import pandas as pd
import os
from docx import Document
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Resume Screening AI",
    layout="wide"
)

st.title("AI-Based Resume Screening and Candidate Recommendation System")

st.write(
    "Upload a job description and rank resumes using NLP and TF-IDF."
)

def read_docx(file_path):
    doc = Document(file_path)

    text = ""

    for para in doc.paragraphs:
        text += para.text + " "

    return text

job_description = st.text_area(
    "Enter Job Description",
    """Python
Machine Learning
SQL
Data Analysis
Pandas
Data Visualization"""
)

if st.button("Analyze Resumes"):

    resume_folder = "resumes"

    resume_texts = []
    resume_names = []

    for file in os.listdir(resume_folder):

        if file.endswith(".docx"):

            path = os.path.join(
                resume_folder,
                file
            )

            text = read_docx(path)

            resume_texts.append(text)
            resume_names.append(file)

    documents = [job_description] + resume_texts

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(
        documents
    )

    scores = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:]
    ).flatten()

    results = pd.DataFrame({
        "Resume": resume_names,
        "Match Score (%)": scores * 100
    })

    results = results.sort_values(
        by="Match Score (%)",
        ascending=False
    )

    st.subheader("Resume Ranking")

    st.dataframe(results)

    top_resume = results.iloc[0]

    st.success(
        f"Best Candidate: {top_resume['Resume']} | "
        f"Score: {top_resume['Match Score (%)']:.2f}%"
    )

    fig, ax = plt.subplots(
        figsize=(10,5)
    )

    top10 = results.head(10)

    ax.bar(
        top10["Resume"],
        top10["Match Score (%)"]
    )

    ax.set_title(
        "Top Resume Scores"
    )

    ax.set_ylabel(
        "Match Score (%)"
    )

    plt.xticks(
        rotation=45
    )

    st.pyplot(fig)  