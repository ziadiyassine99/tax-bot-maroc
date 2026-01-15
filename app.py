"""
IYYA - Assistant Juridique Marocain
Multi-module legal assistant for Moroccan law (CGI, Code du Travail, etc.)
"""

import streamlit as st
from typing import Optional, Tuple, Dict, Any

from config import MODULES, get_module_config
from document_loader import DocumentProcessor, PDFLoadError
from vector_store import VectorStoreManager, create_vector_store_manager
from rag_chain import RAGChainBuilder, RAGQueryHandler, create_rag_chain


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="IYYA - Assistant Juridique",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =============================================================================
# Golden Theme Styling
# =============================================================================

def apply_golden_theme():
    """Apply the golden/beige IYYA theme."""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');
        
        /* Main background - warm beige gradient */
        .stApp {
            background: linear-gradient(180deg, #E8DCC8 0%, #D4C4A8 50%, #C9B896 100%);
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Reduce top padding to move header up */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
        
        /* Main title styling */
        .main-title {
            font-family: 'Playfair Display', serif;
            font-size: 3.5rem;
            font-weight: 700;
            color: #8B6914;
            text-align: center;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        .subtitle {
            font-family: 'Inter', sans-serif;
            color: #6B5A3E;
            text-align: center;
            font-size: 1.2rem;
            font-weight: 400;
            margin-bottom: 0.5rem;
        }
        
        /* Force uniform column heights */
        [data-testid="column"] {
            display: flex;
            flex-direction: column;
        }
        
        [data-testid="column"] > div {
            flex-grow: 1;
            display: flex;
            flex-direction: column;
        }
        
        /* Module cards */
        .module-card {
            background: linear-gradient(135deg, #FFF8EC 0%, #F5EBD7 100%);
            border: 2px solid #D4A574;
            border-radius: 16px;
            padding: 1.5rem;
            margin: 0.75rem 0;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(139, 105, 20, 0.15);
            height: 240px; /* Fixed height for perfect alignment */
            display: flex;
            flex-direction: column;
        }
        
        .module-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(139, 105, 20, 0.25);
            border-color: #B8860B;
        }
        
        .module-icon {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .module-name {
            font-family: 'Playfair Display', serif;
            font-size: 1.2rem;
            font-weight: 600;
            color: #5D4E37;
            margin-bottom: 0.25rem;
            min-height: 3rem;
        }
        
        .module-desc {
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
            color: #7A6B5A;
            flex-grow: 1;
        }
        
        /* Chat styling */
        .stChatMessage {
            background: linear-gradient(135deg, #FFFDF8 0%, #FFF8EC 100%) !important;
            border: 2px solid #C9A86C !important;
            border-radius: 16px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        /* Chat message text - DARK and readable */
        .stChatMessage p, .stChatMessage span, .stChatMessage div {
            color: #2D2A26 !important;
        }
        
        .stChatMessage strong {
            color: #1A1815 !important;
        }
        
        /* Markdown in chat */
        .stMarkdown p {
            color: #2D2A26 !important;
        }
        
        /* Chat input */
        .stChatInput > div {
            background: #FFF8EC !important;
            border: 2px solid #D4A574 !important;
            border-radius: 12px !important;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #D4A574 0%, #B8860B 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 2rem;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(139, 105, 20, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(139, 105, 20, 0.4);
        }
        
        /* Success/Error messages */
        .stSuccess {
            background: linear-gradient(135deg, #D4EDDA 0%, #C3E6CB 100%);
            border-left: 4px solid #28A745;
            border-radius: 8px;
        }
        
        /* Fix green on green - ensure success text is dark */
        .stSuccess p, .stSuccess span, .stSuccess div, .stSuccess svg {
            color: #155724 !important;
            fill: #155724 !important;
        }
        
        .stSuccess [data-testid="stMarkdownContainer"] p {
            color: #155724 !important;
        }
        
        .stError {
            background: linear-gradient(135deg, #F8D7DA 0%, #F5C6CB 100%);
            border-left: 4px solid #DC3545;
            border-radius: 8px;
        }
        
        /* Sidebar */
        .css-1d391kg, [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #F5EBD7 0%, #E8DCC8 100%);
        }
        
        /* Divider */
        hr {
            border-color: #D4A574;
            opacity: 0.5;
        }
        
        /* Section header */
        .section-header {
            font-family: 'Playfair Display', serif;
            font-size: 1.5rem;
            font-weight: 600;
            color: #8B6914;
            text-align: center;
            margin: 0rem 0 1rem 0;
            padding: 0.5rem;
            border-bottom: 2px solid #D4A574;
        }
        
        /* Back button area */
        .back-button {
            margin-bottom: 1rem;
        }
        
        /* Powered by footer */
        .powered-by {
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
            color: #7A6B5A;
            text-align: center;
            margin-top: 2rem;
            padding: 1rem;
        }
        
        .powered-by a {
            color: #8B6914;
            text-decoration: none;
            font-weight: 600;
        }
        
        /* Sources panel styling */
        .sources-container {
            background: linear-gradient(135deg, #F0F4F8 0%, #E8ECF0 100%);
            border: 1px solid #B8C4CE;
            border-radius: 12px;
            padding: 0.75rem 1rem;
            margin-top: 0.5rem;
            font-family: 'Inter', sans-serif;
        }
        
        .sources-header {
            color: #4A5568;
            font-size: 0.9rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
        }
        
        .source-chunk {
            background: #FFFFFF;
            border: 1px solid #CBD5E0;
            border-radius: 8px;
            padding: 0.75rem;
            margin: 0.5rem 0;
            font-size: 0.8rem;
        }
        
        .source-title {
            color: #2D3748;
            font-weight: 600;
            font-size: 0.85rem;
            margin-bottom: 0.25rem;
        }
        
        .source-content {
            color: #4A5568;
            font-size: 0.8rem;
            line-height: 1.4;
            max-height: 100px;
            overflow-y: auto;
        }
        </style>
    """, unsafe_allow_html=True)


# =============================================================================
# Session State Management
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "current_module" not in st.session_state:
        st.session_state.current_module = None
    if "messages" not in st.session_state:
        st.session_state.messages = {}
    if "sources" not in st.session_state:
        st.session_state.sources = {}  # Store sources for each message
    if "module_initialized" not in st.session_state:
        st.session_state.module_initialized = {}


def set_current_module(module_id: str):
    """Set the current active module."""
    st.session_state.current_module = module_id
    if module_id not in st.session_state.messages:
        module_config = get_module_config(module_id)
        st.session_state.messages[module_id] = [
            {
                "role": "assistant",
                "content": f"Bonjour ! Je suis votre assistant spécialisé dans le **{module_config['name']}**. 🇲🇦\n\nPosez-moi vos questions et je vous répondrai en citant les articles pertinents."
            }
        ]


def go_back_to_home():
    """Return to the home page."""
    st.session_state.current_module = None


# =============================================================================
# Cached Resources
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_module_resources(module_id: str) -> Tuple[bool, Optional[str], int, Optional[VectorStoreManager]]:
    """
    Load and cache resources for a specific module.
    
    Returns:
        Tuple of (success, error_message, num_chunks, vector_store_manager)
    """
    try:
        module_config = get_module_config(module_id)
        
        # Create document processor
        doc_processor = DocumentProcessor(pdf_path=module_config["pdf_path"])
        
        # Create vector store manager
        vs_manager = create_vector_store_manager(module_config)
        
        # In memory mode (cloud): always load and create fresh
        if vs_manager.is_memory_mode():
            documents = doc_processor.load_and_split()
            vs_manager.create_vector_store(documents)
            return True, None, len(documents), vs_manager
        
        # Docker mode: check if vector store exists
        if vs_manager.vector_store_exists():
            vs_manager.load_vector_store()
            return True, None, -1, vs_manager
        
        # Load and process documents
        documents = doc_processor.load_and_split()
        
        # Create vector store
        vs_manager.create_vector_store(documents)
        
        return True, None, len(documents), vs_manager
        
    except PDFLoadError as e:
        return False, str(e), 0, None
    except Exception as e:
        return False, f"Erreur inattendue: {str(e)}", 0, None


# Version bump to invalidate cache when prompts change (NO underscore = included in cache key)
# v8: Ajout règle de priorité temporelle - utiliser le décret le plus récent
RAG_CHAIN_VERSION = "v8_temporal_priority"

@st.cache_resource(show_spinner=False)
def get_rag_chain(_vs_manager: VectorStoreManager, module_id: str, version: str = RAG_CHAIN_VERSION) -> RAGChainBuilder:
    """Get or create the RAG chain for a module."""
    module_config = get_module_config(module_id)
    return create_rag_chain(_vs_manager, module_config)


# =============================================================================
# Home Page
# =============================================================================

def render_home_page():
    """Render the home page with module selection."""
    
    # Logo and title
    st.markdown('<h1 class="main-title">⚖️ IYYA</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Votre assistant juridique intelligent<br>basé sur la législation marocaine en vigueur</p>',
        unsafe_allow_html=True
    )
    
    # Section header
    st.markdown('<div class="section-header">Choisissez votre module</div>', unsafe_allow_html=True)
    
    # Module cards in columns
    cols = st.columns(5)
    
    # Iterate over modules and display them in columns
    for idx, (module_id, module_config) in enumerate(MODULES.items()):
        with cols[idx % 5]:
            # Create a card-like button
            card_html = f"""
            <div class="module-card">
                <div class="module-icon">{module_config['icon']}</div>
                <div class="module-name">{module_config['name']}</div>
                <div class="module-desc">{module_config['description']}</div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            
            if st.button(
                f"Accéder au {module_config['short_name']}",
                key=f"btn_{module_id}",
                use_container_width=True
            ):
                set_current_module(module_id)
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div class="powered-by">Powered by <a href="https://wearebeebay.com" target="_blank">wearebeebay</a></div>',
        unsafe_allow_html=True
    )


# =============================================================================
# Chat Page
# =============================================================================

def render_chat_page(module_id: str):
    """Render the chat interface for a specific module."""
    
    module_config = get_module_config(module_id)
    
    # Sidebar with info and back button
    with st.sidebar:
        st.markdown(f"### {module_config['icon']} {module_config['name']}")
        st.markdown(f"_{module_config['description']}_")
        
        st.markdown("---")
        
        if st.button("← Retour à l'accueil", use_container_width=True):
            go_back_to_home()
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("**Exemples de questions :**")
        if module_id == "cgi":
            examples = [
                "Quel est le taux d'IS ?",
                "Exonérations de TVA ?",
                "Régime auto-entrepreneur ?"
            ]
        else:
            examples = [
                "Durée du préavis ?",
                "Calcul des congés payés ?",
                "Indemnité de licenciement ?"
            ]
        for ex in examples:
            st.markdown(f"- _{ex}_")
        
        st.markdown("---")
        st.markdown(
            '<div class="powered-by">Powered by <a href="https://wearebeebay.com">wearebeebay</a></div>',
            unsafe_allow_html=True
        )
    
    # Main content - Back button at the top
    col_back, col_spacer = st.columns([1, 5])
    with col_back:
        if st.button("← Retour", key="main_back_btn"):
            go_back_to_home()
            st.rerun()
    
    st.markdown(
        f'<h1 class="main-title">{module_config["icon"]} {module_config["short_name"]}</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<p class="subtitle">{module_config["name"]}</p>',
        unsafe_allow_html=True
    )
    
    # Load module resources
    with st.spinner(f"🔄 Chargement de la base {module_config['short_name']}..."):
        success, error, num_chunks, vs_manager = load_module_resources(module_id)
    
    # Show status
    if success:
        if num_chunks == -1:
            st.success(f"✅ Base {module_config['short_name']} chargée")
        else:
            st.success(f"✅ Base {module_config['short_name']} créée ({num_chunks} segments)")
    else:
        st.error(f"❌ {error}")
        st.info(f"💡 Vérifiez que '{module_config['pdf_path']}' est présent.")
        st.stop()
    
    # Initialize RAG chain
    rag_chain = get_rag_chain(vs_manager, module_id)
    query_handler = RAGQueryHandler(rag_chain, module_id)
    
    st.markdown("---")
    
    # Chat history - display messages with their sources
    for idx, message in enumerate(st.session_state.messages.get(module_id, [])):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages (except the initial greeting)
            if message["role"] == "assistant" and idx > 0:
                sources_key = f"{module_id}_{idx}"
                if sources_key in st.session_state.sources:
                    sources = st.session_state.sources[sources_key]
                    if sources:
                        with st.expander(f"📚 Sources consultées ({len(sources)} documents)", expanded=False):
                            for i, source in enumerate(sources, 1):
                                doc_name = source.get("source", "Document")
                                page = source.get("page", "N/A")
                                content = source.get("content", "")[:400]
                                if len(source.get("content", "")) > 400:
                                    content += "..."
                                
                                st.markdown(f"""
                                <div class="source-chunk">
                                    <div class="source-title">📄 Source {i} - {doc_name} (Page {page})</div>
                                    <div class="source-content">{content}</div>
                                </div>
                                """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input(f"Posez votre question sur le {module_config['short_name']}..."):
        # Add user message
        st.session_state.messages[module_id].append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response with conversation context (streaming)
        with st.chat_message("assistant"):
            conversation_history = st.session_state.messages.get(module_id, [])
            
            # Stream the response
            response = st.write_stream(query_handler.stream(prompt, conversation_history=conversation_history))
            
            # Get source documents after streaming
            source_docs = query_handler.get_source_documents(prompt, k=5)
            
            # Format sources for display and storage
            formatted_sources = []
            for doc in source_docs:
                formatted_sources.append({
                    "source": doc.metadata.get("source", "Document").split("\\")[-1].split("/")[-1],  # Get filename
                    "page": doc.metadata.get("page", "N/A"),
                    "content": doc.page_content
                })
            
            # Display sources in expander
            if formatted_sources:
                with st.expander(f"📚 Sources consultées ({len(formatted_sources)} documents)", expanded=False):
                    for i, source in enumerate(formatted_sources, 1):
                        content_preview = source["content"][:400]
                        if len(source["content"]) > 400:
                            content_preview += "..."
                        
                        st.markdown(f"""
                        <div class="source-chunk">
                            <div class="source-title">📄 Source {i} - {source['source']} (Page {source['page']})</div>
                            <div class="source-content">{content_preview}</div>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Add assistant message and store sources
        st.session_state.messages[module_id].append({"role": "assistant", "content": response})
        
        # Store sources with message index
        msg_idx = len(st.session_state.messages[module_id]) - 1
        sources_key = f"{module_id}_{msg_idx}"
        st.session_state.sources[sources_key] = formatted_sources
    
    # Clear chat button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🗑️ Nouvelle conversation", use_container_width=True):
            # Clear sources for this module
            keys_to_remove = [k for k in st.session_state.sources if k.startswith(f"{module_id}_")]
            for k in keys_to_remove:
                del st.session_state.sources[k]
            st.session_state.messages[module_id] = [st.session_state.messages[module_id][0]]
            st.rerun()


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    apply_golden_theme()
    init_session_state()
    
    if st.session_state.current_module is None:
        render_home_page()
    else:
        render_chat_page(st.session_state.current_module)


if __name__ == "__main__":
    main()
