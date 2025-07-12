import os
from dotenv import load_dotenv
import streamlit as st
from utils import query_agent_groq, generate_visual, save_chat_history

load_dotenv()
st.set_page_config(page_title="📊 Chart GPT", layout="wide")

st.title("📈 ChartGPT")
st.markdown("Upload CSV/Excel files, ask data questions, get answers + charts and visuals !")

uploaded_files = st.file_uploader(
    "Upload one or more CSV/Excel files",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_fig" not in st.session_state:
    st.session_state.last_fig = None

user_query = st.chat_input("Ask something about your dataset...")

if uploaded_files and user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    response, fig = query_agent_groq(uploaded_files, user_query)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    st.markdown("### 🧠 Answer:")
    st.write(response)

    if fig:
        st.markdown("### 📊 Visual Output:")
        st.pyplot(fig)
        st.session_state.last_fig = fig  # Save latest chart

# Show full chat history and PDF download option
if st.session_state.chat_history:
    with st.expander("💬 Chat History"):
        for msg in st.session_state.chat_history:
            st.markdown(f"**{msg['role'].capitalize()}**: {msg['content']}")

    if st.button("📥 Download PDF Report"):
        report_pdf = save_chat_history(
            st.session_state.chat_history,
            fig=st.session_state.last_fig
        )
        st.download_button("Download PDF", report_pdf, file_name="chat_report.pdf", mime="application/pdf")
