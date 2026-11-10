import os
import sys
import hashlib
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Garandeer dat de 'src' en 'llm' mappen vindbaar zijn
sys.path.append(str(Path(__file__).resolve().parent))

from src.pdf_processor import build_pdf_vectorstore
from llm.gemini_ai_assistant import GeminiAssistant

# Laad omgevingsvariabelen (.env)
load_dotenv()

#------------------------------------------------------------------------------
# Pagina Instellingen & Custom Styling
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="PDF RAG Analyzer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Native Streamlit CSS Styling (Diepblauw Thema zonder JS DOM-injecties)
# Native Streamlit CSS Styling (Professioneel Deep Blue & Slate Thema)
st.markdown("""
<style>
    /* Forceer de hoofdachtergrond naar een diepblauwe gradient */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background: linear-gradient(135deg, #0a192f 0%, #0f172a 100%) !important;
        color: #f8fafc !important;
    }

    /* Sidebar styling: Donkerder marineblauw met een strakke blauwe rand */
    [data-testid="stSidebar"], [data-testid="stSidebarUserContent"] {
        background-color: #0b1329 !important;
        border-right: 1px solid rgba(59, 130, 246, 0.2) !important;
    }

    /* Alle Streamlit basisteksten, labels en titels wit/lichtgrijs maken */
    .stMarkdown, p, h1, h2, h3, h4, h5, h6, label, [data-testid="stWidgetLabel"] p {
        color: #f1f5f9 !important;
    }

    /* Inputs, Selectboxes en Sliders */
    .stTextInput input, .stSelectbox > div > div, div[data-baseweb="select"] {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        border: 1px solid rgba(59, 130, 246, 0.4) !important;
        border-radius: 8px !important;
    }
    
    /* Input focus state */
    .stTextInput input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25) !important;
    }

    /* Professionele Gradient Knoppen met hover effect */
    .stButton>button, .stFormSubmitButton>button {
        background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
        border-color: #60a5fa !important;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.4) !important;
    }

    /* Drag & Drop File Uploader styling */
    [data-testid="stFileUploader"] section {
        background-color: #0f172a !important;
        border: 2px dashed rgba(59, 130, 246, 0.4) !important;
        border-radius: 12px !important;
    }

    /* RAG Blokken / Cards */
    .rag-card {
        background: #1e293b !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 12px !important;
        padding: 18px !important;
        margin-bottom: 14px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25) !important;
    }

    /* Info Metric Boxen bovenin */
    .metric-box {
        background: #111827 !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 10px !important;
        padding: 14px !important;
        text-align: center !important;
        box-shadow: inset 0 0 10px rgba(59, 130, 246, 0.05) !important;
    }

    /* Badges & Labels */
    .badge-pdf {
        background-color: rgba(37, 99, 235, 0.2) !important;
        color: #93c5fd !important;
        border: 1px solid rgba(59, 130, 246, 0.5) !important;
        padding: 4px 12px !important;
        border-radius: 20px !important;
        font-size: 11px !important;
        font-weight: 600 !important;
    }
    
    .badge-serp {
        background-color: rgba(14, 165, 233, 0.2) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(14, 165, 233, 0.5) !important;
        padding: 4px 12px !important;
        border-radius: 20px !important;
        font-size: 11px !important;
        font-weight: 600 !important;
    }

    /* Broncode / Snippets inspectie */
    .source-chunk {
        background-color: #090d16 !important;
        border-left: 4px solid #3b82f6 !important;
        padding: 12px !important;
        border-radius: 8px !important;
        color: #cbd5e1 !important;
        font-family: 'Fira Code', monospace !important;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Sidebar - Instellingen & Parameters
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Parameters")
    
    env_key = os.getenv("GOOGLE_API_KEY") or ""
    api_key = env_key
    
    selected_model = st.selectbox(
        "Gemini LLM Model",
        [ "gemini-3.8-flash",
        "gemini-3.1-pro-preview",],
        help="Selecteer het gewenste Gemini model. Flash modellen zijn geoptimaliseerd voor snelheid."
    )

    st.markdown("---")
    st.markdown("**🔍 Vector & Retrieval Controls**")
    chunk_size = st.slider("Chunk Size (tekstlengte)", 500, 3000, 1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap (overlap)", 0, 500, 200, step=50)
    k_chunks = st.slider("Top-K Chunks Retrieval (k)", 1, 10, 4)

    st.markdown("---")
    enable_serp = st.checkbox("🌐 Google Search Grounding (SERP)", value=True, help="Schakel live webbronnen in voor aanvullende context.")

# ------------------------------------------------------------------------------
# Session State & Caching
# ------------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

@st.cache_resource(show_spinner=False)
def get_cached_vectorstore(file_bytes, key, c_size, c_overlap):
    """
    Haalt een ge-cachete vectorstore op of bouwt deze op voor het geüploade PDF-bestand.
    """
    return build_pdf_vectorstore(file_bytes, key, c_size, c_overlap)

# ------------------------------------------------------------------------------
# Hoofdapplicatie Logic
# ------------------------------------------------------------------------------
st.title("📚 PDF RAG Analyzer")

st.info("""
         🔍 AI-Powered Document Analysis

            Upload a PDF document and ask questions in natural language.

                The system searches the most relevant sections 
                of the document before generating an answer, 
                helping provide responses that are grounded in 
                the actual contents of your PDF.
                   """)

uploaded_file = st.file_uploader(
    "Upload your PDF file to analyze",
    type=["pdf"]
)

if uploaded_file:
    if not api_key:
        st.warning("⚠️ Please enter your Google API Key in the sidebar.")
    else:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        with st.spinner('Scanning PDF and building CHROMA vector store...'):
            try:
                # Geef de raw file bytes door aan de cached loader
                vectorstore, total_pages, total_chunks = get_cached_vectorstore(
                    file_bytes,
                    api_key,
                    chunk_size,
                    chunk_overlap
                )
                
                # Metrics Banner Ribbon
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">Document</div>
                        <div style="font-size: 13px; font-weight: 600; color: #f8fafc; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{uploaded_file.name}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">Total Pages</div>
                        <div style="font-size: 14px; font-weight: 600; color: #38bdf8;">{total_pages} Pages</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">Vector Chunks</div>
                        <div style="font-size: 14px; font-weight: 600; color: #818cf8;">{total_chunks} Chunks</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">CHROMA Status</div>
                        <div style="font-size: 14px; font-weight: 600; color: #4ade80;">Ready (In-Memory)</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                import traceback
                st.error(f"An error occurred while processing the PDF: {e}")
                st.text(traceback.format_exc())
                st.stop()

        st.markdown("<br>", unsafe_allow_html=True)
        
        # ------------------------------------------------------------------------------
        # Workspace Grid: Chat Interface (Links) & Context Inspector (Rechts)
        # ------------------------------------------------------------------------------
        chat_col, inspector_col = st.columns([1.8, 1.2])

        # Linkerkolom: Chat sessie
        with chat_col:
            st.markdown("### 💬 Interactive Q&A Session")
            
            with st.form("query_form", clear_on_submit=True):
                user_query = st.text_input("Ask your question:", placeholder="E.g. What were the net revenues in 2025?")
                submit_button = st.form_submit_button("Search & Generate Answer ➔")
                
                if submit_button and user_query:
                    try:
                        docs = vectorstore.similarity_search(user_query, k=k_chunks)

                        print("\n" + "=" * 80)
                        print("RETRIEVED CHUNKS")
                        print("=" * 80)

                        for i, doc in enumerate(docs, start=1):
                            print(f"\nCHUNK {i}")
                            print(doc.page_content[:1000])

                        print("=" * 80)
                        
                        context_chunks = [d.page_content for d in docs]
                        context_text = "\n\n".join(context_chunks)

                        assistant = GeminiAssistant(
                            api_key=api_key,      
                           
                        )

                        answer, web_sources = assistant.ask_with_context(
                            user_query, 
                            context_text,
                            enable_serp=enable_serp
                        )
                        
                        st.session_state.chat_history.append({
                            "query": user_query,
                            "answer": answer,
                            "model": selected_model,
                            "docs": docs,
                            "web_sources": web_sources if web_sources else []
                        })
                    except Exception as e:
                        st.error(f"Error bij het genereren van het antwoord: {e}")

            for entry in reversed(st.session_state.chat_history):

                st.markdown(f"""
                <div class="rag-card">
                    <div style="
                        font-size:16px;
                        font-weight:600;
                        color:#818cf8;
                        margin-bottom:12px;
                    ">
                        ❓ Vraag: {entry['query']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.write(entry["answer"])

                st.markdown(f"""
                <div style="margin-bottom:20px;">
                    <span class="badge-pdf">
                        📄 PDF Context: {len(entry['docs'])} Chunks used
                    </span>
                </div>
                """, unsafe_allow_html=True)
                 
        # Rechterkolom: Context Inspector Cards
        with inspector_col:
            st.markdown("### 🔍 Context & Source Inspector")
            
            chunk_preview_length = 300
            
            if st.session_state.chat_history:
                last_entry = st.session_state.chat_history[-1]
                
                st.markdown(f"<div style='font-size:12px; color:#94a3b8; margin-bottom:10px;'>Last Query: <b style='color:#f8fafc;'>\"{last_entry['query']}\"</b></div>", unsafe_allow_html=True)
                
                for i, doc in enumerate(last_entry['docs']):
                    page_num = doc.metadata.get('page', 0) + 1
                    st.markdown(f"""
                    
                        <div class="source-chunk">
                            {doc.page_content[:chunk_preview_length]}...
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                if last_entry.get('web_sources'):
                    st.markdown("**🌐 Google SERP Search Sources:**")
                    for src in last_entry.get('web_sources', []):
                        st.markdown(f"- {src}")
            else:
                st.info("Ask a question to inspect the retrieved PDF chunks and external sources here.")