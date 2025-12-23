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
    LLM_MODEL: str = "gpt-4o-mini"
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
        "system_prompt": """Tu es un expert en sécurité sociale marocaine (CNSS). Réponds à partir du CONTEXTE fourni.

INSTRUCTIONS :
1. Réponds TOUJOURS dans la langue de la question (français, arabe, anglais, etc.)
2. Base ta réponse sur le CONTEXTE ci-dessous
3. Cite les sources si disponibles dans le contexte
4. Sois COMPLET et PRÉCIS
5. Si le contexte ne contient pas l'information, dis-le clairement

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
        "system_prompt": """Tu es un expert en conventions internationales et traités du Maroc. Réponds à partir du CONTEXTE fourni.

INSTRUCTIONS :
1. Réponds TOUJOURS dans la langue de la question.
2. Base ta réponse EXCLUSIVEMENT sur le CONTEXTE fourni.
3. Cite les articles et traités pertinents.
4. Si le contexte ne répond pas à la question, indique-le.

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
        "system_prompt": """Tu es un expert en régulation financière marocaine. Réponds à partir du CONTEXTE fourni.

INSTRUCTIONS :
1. Réponds TOUJOURS dans la langue de la question.
2. Utilise le CONTEXTE pour formuler ta réponse.
3. Sois précis sur les textes de loi et règlements.
4. Indique si l'information est absente du contexte.

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
        "system_prompt": """Tu es un expert en droit du travail marocain. Réponds à partir du CONTEXTE fourni.

INSTRUCTIONS :
1. Réponds TOUJOURS dans la langue de la question.
2. Base ta réponse sur le CONTEXTE ci-dessous (Code du Travail, etc.).
3. Cite les articles pertinents (ex: "Article X").
4. Si l'information n'est pas dans le contexte, dis-le.

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
