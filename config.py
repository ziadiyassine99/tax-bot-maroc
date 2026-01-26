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
    CHUNK_SIZE: int = 2500
    CHUNK_OVERLAP: int = 500


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for OpenAI models."""
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4o"
    LLM_TEMPERATURE: float = 0.3


# =============================================================================
# MODULE CONFIGURATIONS
# =============================================================================

MODULES: Dict[str, Dict[str, Any]] = {
    "cnss": {
        "id": "cnss",
        "name": "CNSS",
        "short_name": "CNSS",
        "description": "Sécurité sociale et prestations",
        "pdf_path": "documents/CNSS_IYYA",
        "collection_name": "cnss_iyya_docs",
        "icon": "🏥",
        "color": "#D4A574",
        "system_prompt": """Tu es un expert en sécurité sociale marocaine (CNSS, AMO, régimes de retraite).

RÈGLES :
1. Cite les articles : "En vertu de l'article X du [décret/loi], ..."
2. Taux EXACTS (ex: 4,52%, pas d'arrondi)
3. Ne jamais inventer de taux ou d'article
4. Si information absente : "Cette information n'est pas présente dans les documents fournis."

CONTEXTE :
{context}

QUESTION : {question}

RÉPONSE :"""
    },
    "conventions": {
        "id": "conventions",
        "name": "Conventions Internationales",
        "short_name": "Conventions",
        "description": "Traités et accords internationaux",
        "pdf_path": "documents/Conventions internationales_IYYA",
        "collection_name": "conventions_iyya_docs",
        "icon": "🌍",
        "color": "#8B7355",
        "system_prompt": """Tu es un expert en conventions internationales et traités ratifiés par le Maroc.

RÈGLES :
1. Cite les articles : "En vertu de l'article X de la Convention/Traité Y, ..."
2. Mentionne les pays signataires si pertinent
3. Ne jamais inventer d'articles ou de dispositions
4. Si information absente : "Cette information n'est pas présente dans les documents fournis."

CONTEXTE :
{context}

QUESTION : {question}

RÉPONSE :"""
    },
    "regulation": {
        "id": "regulation",
        "name": "Régulation Financière",
        "short_name": "Régulation",
        "description": "Droit bancaire et marché des capitaux",
        "pdf_path": "documents/Regulation financiere_IYYA",
        "collection_name": "regulation_iyya_docs",
        "icon": "🏦",
        "color": "#A89F91",
        "system_prompt": """Tu es un expert en régulation financière marocaine (banques, assurances, marchés des capitaux, IFRS).

RÈGLES :
1. Cite les articles : "En vertu de l'article X de l'Instruction/Circulaire Y, ..."
2. Dates précises d'application
3. Ne jamais inventer de dispositions ou de dates
4. Si information absente : "Cette information n'est pas présente dans les documents fournis."

CONTEXTE :
{context}

QUESTION : {question}

RÉPONSE :"""
    },
    "travail": {
        "id": "travail",
        "name": "Travail",
        "short_name": "Travail",
        "description": "Contrats, licenciement et congés",
        "pdf_path": "documents/TRAVAIL_IYYA",
        "collection_name": "travail_iyya_docs",
        "icon": "👷",
        "color": "#C4A484",
        "system_prompt": """Tu es un expert en droit du travail marocain (Code du Travail, décrets d'application, SMIG/SMAG).

RÈGLES :
1. Cite les articles : "En vertu de l'article X du Code du Travail, ..."
2. Montants et durées EXACTS
3. Utilise les montants les plus RÉCENTS (décrets récents remplacent les anciens)
4. Ne jamais inventer de chiffres ou d'articles
5. Si information absente : "Cette information n'est pas présente dans les documents fournis."

CONTEXTE :
{context}

QUESTION : {question}

RÉPONSE :"""
    },
    "cgi": {
        "id": "cgi",
        "name": "Code Général des Impôts",
        "short_name": "CGI",
        "description": "Fiscalité et impôts marocains",
        "pdf_path": "documents/CGI_IYYA",
        "collection_name": "cgi_iyya_docs",
        "icon": "💰",
        "color": "#B8860B",
        "system_prompt": """Tu es un expert en fiscalité marocaine (Code Général des Impôts, IS, IR, TVA, droits d'enregistrement).

RÈGLES :
1. Cite les articles : "En vertu de l'article X du CGI, ..."
2. Taux EXACTS (pas d'arrondi)
3. Ne jamais inventer de taux ou d'articles
4. Si information absente : "Cette information n'est pas présente dans les documents fournis."

CONTEXTE :
{context}

QUESTION : {question}

RÉPONSE :"""
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


# =============================================================================
# QDRANT CLOUD CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class QdrantConfig:
    """Configuration for Qdrant Cloud."""
    URL: str = "https://039bd739-4648-44a6-b028-2cd2fd0a8dcb.us-east4-0.gcp.cloud.qdrant.io"
    API_KEY: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.PScjOF-OCevSmIL5UOM-qq7O8_PSye46d_C6RLRd8HI"


def get_qdrant_config() -> dict:
    """
    Get Qdrant connection configuration.
    Checks environment variables first, then falls back to defaults.
    
    Returns:
        Dict with 'url' and 'api_key' keys
    """
    url = os.getenv("QDRANT_URL", QdrantConfig.URL)
    api_key = os.getenv("QDRANT_API_KEY", QdrantConfig.API_KEY)
    
    return {
        "url": url,
        "api_key": api_key
    }