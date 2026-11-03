import os
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Garandeer dat de 'src' en 'llm' mappen vindbaar zijn
sys.path.append(str(Path(__file__).resolve().parent))

from src.pdf_processor import build_pdf_vectorstore
from llm.gemini_ai_assistant import GeminiAssistant

# Laad omgevingsvariabelen (.env)
load_dotenv()

# ------------------------------------------------------------------------------
# Pagina & Theme Instellingen
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="PDF RAG Analyzer Pro",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS injecteren voor de exacte professionele dark UI styling
st.markdown("""
<style>
    /* Google Fonts & Base Colors */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Achtergrond en hoofdkleuren */
    .stApp {
        background-color: #090d16;
        color: #f8fafc;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }

    /* Dark Glassmorphism Cards */
    .rag-card {
        background-color: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        backdrop-filter: blur(12px);
    }
    
    .metric-box {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 12px;
        text-align: center;
    }

    /* Custom Input & Textarea */
    .stTextInput input, .stSelectbox select {
        background-color: #090d16 !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    
    /* Badges & Pills */
    .badge-pdf {
        background-color: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 500;
    }
    
    .badge-serp {
        background-color: rgba(14, 165, 233, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(14, 165, 233, 0.3);
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 500;
    }

    /* Bronnenweergave codeblokken */
    .source-chunk {
        background-color: #090d16;
        border-left: 3px solid #6366f1;
        padding: 10px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #cbd5e1;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Sidebar - Instellingen & Parameters
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Parameters")
    
    # Sleutel automatisch ophalen of via UI inschrijven
    env_key = os.getenv("GOOGLE_API_KEY") or ""
    api_key = st.text_input("Gemini API Key", value=env_key, type="password", help="Sleutel uit .env of handmatig invullen")
    
    st.markdown("---")
    selected_model = st.selectbox(
        "Gemini LLM Model",
        ["models/gemini-2.5-flash", "models/gemini-1.5-pro", "models/gemini-1.5-flash"],
        help="Select the desired Gemini model for " \
        "response generation. Flash models are optimized for speed and cost."
    )

    #Input for pdf chunking and retrieval parameters
    st.markdown("---")
    st.markdown("**🔍 Vector & Retrieval Controls**")
    chunk_size = st.slider("Chunk Size (text length)", 500, 3000, 1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap (overlap)", 0, 500, 200, step=50)
    k_chunks = st.slider("Top-K Chunks Retrieval (k)", 1, 10, 4)

    st.markdown("---")
    enable_serp = st.checkbox("🌐 Google Search Grounding (SERP)", value=True, help="Schakel live webbronnen in voor aanvullende context.")

# ------------------------------------------------------------------------------
# Session State & Caching
# ------------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

import hashlib

@st.cache_resource(show_spinner=False)
def get_cached_vectorstore(file_name, file_size, file_bytes, key, c_size, c_overlap):
    # Optionally, use a hash of the file for even more robust caching
    file_hash = hashlib.md5(file_bytes).hexdigest()
    return build_pdf_vectorstore(file_bytes, key, c_size, c_overlap)

# ------------------------------------------------------------------------------
# Header & Dashboard Title
# ------------------------------------------------------------------------------
st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
    <div>
        <h1 style="margin: 0; font-size: 26px; font-weight: 700; background: linear-gradient(to right, #ffffff, #94a3b8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            📚 PDF RAG Analyzer Pro
        </h1>
        <p style="margin: 4px 0 0 0; font-size: 13px; color: #64748b;">
            Document AI Studio powered by FAISS Vector Indexing & Google Gemini
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Main Layout: PDF Upload & Metrics
# ------------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file:
    if not api_key:
        st.warning("⚠️ Please enter your Google API Key in the sidebar.")
    else:
        with st.spinner('Scanning PDF and building FAISS vector store...'):
            try:
                vectorstore, total_pages, total_chunks = get_cached_vectorstore(
                    uploaded_file.name,
                    uploaded_file.size,
                    uploaded_file.getvalue(),
                    api_key,
                    chunk_size,
                    chunk_overlap
                )
                
                # Metrics Banner Ribbon
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div style="font-size: 11px; color: #94a3b8;">Document</div>
                        <div style="font-size: 13px; font-weight: 600; color: #f8fafc; overflow: hidden; text-overflow: ellipsis;">{uploaded_file.name}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div style="font-size: 11px; color: #94a3b8;">Total Pages</div>
                        <div style="font-size: 14px; font-weight: 600; color: #38bdf8;">{total_pages} Pages</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div style="font-size: 11px; color: #94a3b8;">Vector Chunks</div>
                        <div style="font-size: 14px; font-weight: 600; color: #818cf8;">{total_chunks} Chunks</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div style="font-size: 11px; color: #94a3b8;">FAISS Status</div>
                        <div style="font-size: 14px; font-weight: 600; color: #4ade80;">Ready (In-Memory)</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"An error occurred while processing the PDF: {e}")
                st.stop()

        st.markdown("<br>", unsafe_allow_html=True)
        
        # ------------------------------------------------------------------------------
        # Workspace Grid: Chat Interface (Links) & Context Inspector (Rechts)
        # ------------------------------------------------------------------------------
        chat_col, inspector_col = st.columns([1.8, 1.2])

        with chat_col:
            st.markdown("### 💬 Interactive Q&A Session")
            
            # Formulier voor vraaginvoer

            with st.form("query_form", clear_on_submit=True):
                user_query = st.text_input("Ask your question:", placeholder="E.g. What were the net revenues in 2025?")
                submit_button = st.form_submit_button("Search & Generate Answer")
                
                if submit_button and user_query:
                    try:
                        # 1. Similarity search in FAISS
                        docs = vectorstore.similarity_search(user_query, k=k_chunks)
                        context_chunks = []
                        for d in docs:
                            page_num = d.metadata.get('page', 0) + 1
                            context_chunks.append(f"--- Pagina {page_num} ---\n{d.page_content}")
                        context_text = "\n\n".join(context_chunks)
                        assistant = GeminiAssistant(
                            api_key=api_key,
                            model=selected_model,
                            system_prompt='''You are a highly intelligent AI assistant 
                            that provides accurate and concise answers based on the 
                            provided context. If the answer is not in the context, 
                            respond with I don't know.'''
                        )

                        # 3. Antwoord genereren (puur op basis van PDF context)
                        answer, web_sources = assistant.generate_response(user_query, context_text)
                        st.session_state.chat_history.append({
                            "query": user_query,
                            "answer": answer,
                            "model": assistant.model,
                            "docs": docs,
                            "web_sources": web_sources if web_sources else []
                        })
                    except Exception as e:
                        st.error(f"Error bij het genereren van het antwoord: {e}")

            # Chat History Weergave
            for entry in reversed(st.session_state.chat_history):
                st.markdown(f"""
                <div class="rag-card">
                    <div style="font-size: 12px; font-weight: 600; color: #818cf8; margin-bottom: 6px;">
                        ❓ Vraag: {entry['query']}
                    </div>
                    <div style="font-size: 13px; color: #e2e8f0; line-height: 1.6; margin-bottom: 10px;">
                        {entry['answer']}
                    </div>
                    <div style="display: flex; gap: 8px; flex-wrap: wrap; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 8px;">
                        <span class="badge-pdf">📄 PDF Context: {len(entry['docs'])} Chunks used</span>
                        {"<span class='badge-serp'>🌐 Live Web Grounded</span>" if entry['web_sources'] else ""}
                        <span style="font-size: 11px; color: #64748b; margin-left: auto;">Model: {entry['model']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
            # Rechterkolom: Context Inspector Cards
            with inspector_col:
                st.markdown("### 🔍 Context & Source Inspector")
                
                if st.session_state.chat_history:
                    last_entry = st.session_state.chat_history[-1]
                    
                    st.markdown(f"Last Query: *\"{last_entry['query']}\"*")
                    
                    for i, doc in enumerate(last_entry['docs']):
                        page_num = doc.metadata.get('page', 0) + 1
                        st.markdown(f"""
                        <div style="background-color: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 10px; margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; font-size: 11px; color: #818cf8; font-weight: 600; margin-bottom: 4px;">
                                <span>Chunk #{i+1} (Pagina {page_num})</span>
                            </div>
                            <div class="source-chunk">
                                {doc.page_content[:300]}...
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    if last_entry.get('web_sources'):
                        st.markdown("**🌐 Google SERP Search :**")
                        for src in last_entry.get('web_sources', []):
                            st.markdown(f"- {src}")
                else:
                    st.info("Ask a question to see the context and sources here.") 
