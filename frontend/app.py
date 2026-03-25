import streamlit as st
import requests
import os
import uuid

# ================= CONFIG =================
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Autonomous Energy Researcher Agent",
    layout="centered",
    page_icon="⚡"
)

# ================= SESSION THREAD ID =================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ================= LIGHT GEN-Z ENERGY THEME =================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e0f7fa, #fce4ec);
    min-height: 100vh;
}

.block-container {
    padding-top: 3rem;
    padding-bottom: 3rem;
    max-width: 700px;
}

h1 {
    text-align: center;
    font-size: 2.5rem;
    font-weight: 700;
    color: #1a237e;
    margin-bottom: 0.5rem;
}

.tagline {
    text-align: center;
    font-size: 1.1rem;
    color: #37474f;
    margin-bottom: 2rem;
}

.card {
    background: white;
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
}

.stTextArea textarea {
    border-radius: 14px;
    border: 2px solid #b2dfdb;
    padding: 14px;
    font-size: 16px;
}

.stButton>button {
    background: linear-gradient(90deg, #26c6da, #ec407a);
    color: white;
    font-weight: 600;
    border-radius: 14px;
    height: 3em;
    border: none;
    width: 100%;
    transition: 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.03);
    opacity: 0.9;
}

.result-box {
    margin-top: 25px;
    padding: 20px;
    background-color: #ffffff;
    border-radius: 15px;
    border: 1px solid #eeeeee;
    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("<h1>⚡ Autonomous Energy Researcher Agent</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='tagline'>Save Energy. Think Smart. Build Sustainable Futures.</div>",
    unsafe_allow_html=True
)

# ================= MAIN CARD =================
st.markdown("<div class='card'>", unsafe_allow_html=True)

query = st.text_area("Enter your energy research topic", height=140)

if st.button("Generate Insight"):
    if not query.strip():
        st.warning("Please enter a topic first.")
    else:
        with st.spinner("Analyzing global energy signals..."):
            try:
                response = requests.post(
                    f"{API_URL}/research",
                    json={
                        "query": query,
                        "thread_id": st.session_state.thread_id
                    }
                )

                # Debug if needed
                if response.status_code != 200:
                    st.error(f"Backend Error: {response.status_code}")
                    st.write(response.text)
                else:
                    data = response.json()

                    st.markdown(
                        f"""
                        <div class="result-box">
                        <h3>📄 Energy Intelligence Report</h3>
                        <p><strong>Topic:</strong> {data.get("query", query)}</p>
                        <hr>
                        <p>{data.get("result", "No result returned.")}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    if data.get("suggestions"):
                        st.markdown("### 💡 Explore Further")
                        for suggestion in data["suggestions"]:
                            st.markdown(f"- {suggestion}")

            except Exception as e:
                st.error(f"Connection error: {e}")

st.markdown("</div>", unsafe_allow_html=True)