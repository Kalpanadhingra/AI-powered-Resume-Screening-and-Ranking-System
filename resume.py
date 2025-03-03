import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------- Styling --------------------------
st.markdown(
    """
    <style>
    /* Set Background to Clean White */
    .stApp {
        background-color: #FFF8E8;
    }

    /* Main Title */
    .main-title {
        font-size: 35px;
        font-weight: bold;
        text-align: center;
        color: #2C3E50;
        padding: 10px;
    }

    /* Sub Title */
    .sub-title {
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        color: #34495E;
        padding: 10px;
    }

    /* Resume Ranking Card */
    .resume-card {
        padding: 15px;
        background: #FFFFFF;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px;
    }

    /* Progress Bar Styling */
    .stProgress > div > div > div > div {
        background-color: #3498DB !important;
    }

    /* Sidebar Styling */
    .stSidebar {
        background-color: #D8BFD8;
        padding: 10px;
        border-radius: 10px;
    }

    /* Sidebar Text Color */
    .css-1d391kg {
        background-color: #2C3E50 !important;
        color: white !important;
    }

    /* Sidebar Text */
    .sidebar-title {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        color: white;
        padding-bottom: 10px;
    }

    /* CSV Download Button */
    .download-button {
        display: block;
        text-align: center;
        padding: 12px;
        font-size: 18px;
        font-weight: bold;
        color: white;
        background-color: #FFd470;
        border-radius: 10px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------- Sidebar --------------------------
with st.sidebar:
    st.markdown('<h2 class="sidebar-title">⚡ Quick Actions</h2>', unsafe_allow_html=True)
    
    # Upload Resume Button
    uploaded_files = st.file_uploader("📤 Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

    # Job Description Input
    job_description = st.text_area("📝 Enter Job Description")

    # Information Links
    st.markdown("📌 **Guidelines:**")
    st.markdown("[🔗 How to Write a Resume?](https://www.resume.com/)")

    # About Section
    st.markdown("---")
    st.markdown("👨‍💻 **Developed by:** Kalpana")

# -------------------------- Main Content --------------------------
st.markdown('<h1 class="main-title">📄 AI Resume Screening & Ranking System</h1>', unsafe_allow_html=True)

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Function to Rank Resumes
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()

    return cosine_similarities

if uploaded_files and job_description:
    st.markdown('<h2 class="sub-title">🏆 Ranking Resumes</h2>', unsafe_allow_html=True)
    
    resumes = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)

    # Rank resumes
    scores = rank_resumes(job_description, resumes)

    # Display scores
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores })
    results = results.sort_values(by="Score", ascending=False)

    # Display results
    for i, row in results.iterrows():
        st.markdown(
            f"""
            <div class="resume-card">
                <h3>📑 {row['Resume']}</h3>
                <p>Matching Score: <b>{row['Score']}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(row["Score"] / 100)

    # Show best match
    top_resume = results.iloc[0]
    st.success(f"🎯 Best Match: **{top_resume['Resume']}** with **{top_resume['Score']}% match!**")

    # Bar Chart Visualization
    st.markdown('<h2 class="sub-title">📊 Resume Ranking Chart</h2>', unsafe_allow_html=True)
    st.bar_chart(results.set_index("Resume"))

    # CSV Download Button
    import base64
    csv = results.to_csv(index=False).encode('utf-8')
    st.markdown(
        f'<a href="data:file/csv;base64,{csv.decode()}" download="resume_ranking.csv" class="download-button">📥 Download Ranking Report</a>',
        unsafe_allow_html=True,
    )
