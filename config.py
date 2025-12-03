"""
Configuration module for the multi-module legal assistant application.
Supports multiple legal documents: CGI (taxes), Code du Travail, etc.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class ChunkingConfig:
    """Configuration for document chunking."""
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 300


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for OpenAI models."""
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-5-mini"
    LLM_TEMPERATURE: float = 0.2  # Lower for more precise, fact-based responses


# =============================================================================
# MODULE CONFIGURATIONS
# =============================================================================

MODULES: Dict[str, Dict[str, Any]] = {
    "cgi": {
        "id": "cgi",
        "name": "Code Général des Impôts",
        "short_name": "CGI",
        "description": "Fiscalité marocaine, IS, IR, TVA, taxes et impôts",
        "pdf_path": "cgi_maroc.pdf",
        "persist_directory": "./chroma_db_cgi",
        "collection_name": "cgi_maroc_docs",
        "icon": "💰",
        "color": "#D4A574",
        "system_prompt": """Tu es un assistant juridique basé UNIQUEMENT sur le Code Général des Impôts du Maroc.

RÈGLES STRICTES :
- Base ta réponse EXCLUSIVEMENT sur le contexte fourni ci-dessous
- N'invente RIEN - si l'info n'est pas dans le contexte, dis-le
- Cite OBLIGATOIREMENT les articles : "Article X du CGI : [contenu]"
- Donne les taux, montants et conditions EXACTS du contexte
- Pas de "Bonjour" ni de "N'hésitez pas"
- Structure avec sections numérotées
- Si hors-sujet fiscal, refuse poliment

CONTEXTE DU CGI (source unique de vérité) :
{context}

Question : {question}

Réponse basée UNIQUEMENT sur le contexte ci-dessus :
"""
    },
    "cdt": {
        "id": "cdt",
        "name": "Code du Travail",
        "short_name": "CDT",
        "description": "Droit du travail marocain, contrats, licenciement, congés",
        "pdf_path": "cdt_maroc.pdf",
        "persist_directory": "./chroma_db_cdt",
        "collection_name": "cdt_maroc_docs",
        "icon": "👷",
        "color": "#8B7355",
        "system_prompt": """Tu es un assistant juridique basé UNIQUEMENT sur le Code du Travail du Maroc.

RÈGLES STRICTES :
- Base ta réponse EXCLUSIVEMENT sur le contexte fourni ci-dessous
- N'invente RIEN - si l'info n'est pas dans le contexte, dis-le
- Cite OBLIGATOIREMENT les articles : "Article X du Code du Travail : [contenu]"
- Donne les durées, délais et montants EXACTS du contexte
- Pas de "Bonjour" ni de "N'hésitez pas"
- Structure avec sections numérotées
- Si hors-sujet droit du travail, refuse poliment

CONTEXTE DU CODE DU TRAVAIL (source unique de vérité) :
{context}

Question : {question}

Réponse basée UNIQUEMENT sur le contexte ci-dessus :
"""
    }
}


def get_module_config(module_id: str) -> Dict[str, Any]:
    """
    Get configuration for a specific module.
    
    Args:
        module_id: The module identifier (e.g., 'cgi', 'cdt')
        
    Returns:
        Dict containing module configuration
        
    Raises:
        ValueError: If module_id is not found
    """
    if module_id not in MODULES:
        raise ValueError(f"Module '{module_id}' not found. Available: {list(MODULES.keys())}")
    return MODULES[module_id]


def get_openai_api_key() -> str:
    """
    Retrieve OpenAI API key from environment or Streamlit secrets.
    
    Returns:
        str: The OpenAI API key
        
    Raises:
        ValueError: If no API key is found
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key:
        return api_key
    
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
            return st.secrets['OPENAI_API_KEY']
    except Exception:
        pass
    
    raise ValueError(
        "Clé API OpenAI non trouvée. "
        "Définissez OPENAI_API_KEY dans les variables d'environnement "
        "ou dans .streamlit/secrets.toml"
    )
