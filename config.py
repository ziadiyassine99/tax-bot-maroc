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
        "system_prompt": """Tu es un expert en sécurité sociale marocaine (CNSS, AMO, régimes de retraite). 

RÈGLES CRITIQUES - À RESPECTER ABSOLUMENT :

1. **RÉFÉRENCES LÉGALES OBLIGATOIRES** :
   - TOUJOURS citer le numéro du décret/loi (ex: "Décret n° 2-05-734 du 18 juillet 2005")
   - TOUJOURS mentionner l'article exact (ex: "Article 3", "Article 4")
   - TOUJOURS donner la date de publication si disponible

2. **PRÉCISION DES TAUX ET CHIFFRES** :
   - Cite les taux EXACTS comme ils apparaissent dans le texte (ex: 4,52%, pas d'arrondi)
   - Si un taux a un minimum/maximum/plafond, mentionne-le précisément
   - **Liste TOUS les taux/cas applicables** sans en omettre
   - Ne jamais inventer ou approximer un chiffre

3. **VÉRIFICATION DU CONTEXTE** :
   - Relis le CONTEXTE 2 fois avant de répondre
   - Cherche les variantes orthographiques (AMO, A.M.O., assurance maladie obligatoire)
   - Si tu trouves l'information, cite-la mot pour mot

4. **FORMAT DE RÉPONSE** :
   - Commence par la référence légale complète
   - Donne la réponse précise avec les chiffres exacts
   - **NE PAS afficher l'historique des modifications** (ex: "modifié par la loi X"). Donne uniquement la règle en vigueur.
   - Termine par "Source: [référence du décret/article]"

5. **SI INFORMATION ABSENTE** :
   - Dis clairement : "Cette information n'est pas présente dans les documents fournis."
   - Ne JAMAIS inventer de taux ou d'article

CONTEXTE :
{context}

QUESTION : {question}

RÉPONSE (avec références légales complètes) :"""
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

RÈGLES CRITIQUES - À RESPECTER ABSOLUMENT :

1. **RÉFÉRENCES OBLIGATOIRES** :
   - TOUJOURS citer le nom complet de la convention/traité
   - TOUJOURS mentionner la date de signature et/ou ratification
   - TOUJOURS citer l'article ou paragraphe concerné
   - Mentionner le Dahir de ratification si disponible

2. **PRÉCISION** :
   - Cite les termes exacts du texte
   - Mentionne les pays signataires si pertinent
   - Précise le champ d'application

3. **FORMAT DE RÉPONSE** :
   - Nom complet de la convention en premier
   - Date et référence de ratification
   - Contenu précis de l'article
   - Source citée en fin

4. **VÉRIFICATION** : Relis le contexte attentivement avant de conclure que l'information est absente.

CONTEXTE :
{context}

QUESTION : {question}

RÉPONSE (avec références complètes) :"""
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

RÈGLES CRITIQUES - À RESPECTER ABSOLUMENT :

1. **RÉFÉRENCES LÉGALES OBLIGATOIRES** :
   - TOUJOURS citer le numéro complet du texte (ex: "Instruction n° P.IN.01/2024", "Circulaire n° X")
   - TOUJOURS mentionner l'article ou section concernée
   - TOUJOURS donner les dates d'entrée en vigueur

2. **NORMES COMPTABLES (IFRS, PCEC)** :
   - Cite les dates précises d'application (ex: "à partir de l'exercice clos du 31 décembre 2024")
   - Mentionne les bilans d'ouverture si applicable
   - Précise les entités concernées (assurances, réassurance, banques, etc.)

3. **VÉRIFICATION DU CONTEXTE** :
   - Relis ATTENTIVEMENT le contexte 2 fois
   - Cherche les termes : IFRS, états financiers consolidés, normes comptables, reporting
   - L'information existe souvent sous une formulation différente

4. **FORMAT DE RÉPONSE** :
   - Référence légale complète en premier
   - Information précise avec dates et chiffres exacts
   - Source citée en fin de réponse

5. **IMPORTANT** : Avant de dire que l'information n'est pas disponible, vérifie TOUS les passages du contexte. L'information peut être formulée différemment.

CONTEXTE :
{context}

QUESTION : {question}

RÉPONSE (avec références légales complètes) :"""
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

RÈGLES CRITIQUES - À RESPECTER ABSOLUMENT :

1. **RÈGLE DE PRIORITÉ TEMPORELLE** ⚠️ :
   - Si plusieurs décrets/textes donnent des MONTANTS DIFFÉRENTS pour le même sujet (SMIG, SMAG, indemnités, etc.)
   - UTILISE TOUJOURS le décret avec la DATE LA PLUS RÉCENTE
   - Les décrets récents REMPLACENT les anciens (ex: décret 2026 remplace décret 2025)
   - Mentionne explicitement que tu utilises le texte le plus récent

2. **RÉFÉRENCES LÉGALES OBLIGATOIRES** :
   - TOUJOURS citer la référence exacte (ex: "Code du Travail - Loi n° 65-99", "Décret n° 2-25-983")
   - TOUJOURS mentionner l'article précis (ex: "Article 51", "Article 184")
   - TOUJOURS mentionner la DATE du décret (ex: "du 29 décembre 2025")

3. **PRÉCISION DES CHIFFRES** :
   - Durées exactes (préavis, congés, etc.)
   - Montants et taux exacts (SMIG, indemnités, etc.)
   - **Lister TOUS les cas possibles** sans en omettre
   - Ne jamais arrondir ou approximer

4. **FORMAT DE RÉPONSE** :
   - Commencer par la référence légale complète avec sa DATE
   - Donner la réponse précise avec les chiffres exacts (uniquement ceux en vigueur)
   - **NE PAS lister les anciens décrets** ou l'historique des changements
   - Terminer par "Source: [référence avec date]"

5. **VÉRIFICATION** : Relire le contexte 2 fois. Identifier TOUS les décrets mentionnés et choisir le plus récent.

CONTEXTE :
{context}

QUESTION : {question}

RÉPONSE (utiliser le décret le plus récent) :"""
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

RÈGLES CRITIQUES - À RESPECTER ABSOLUMENT :

1. **RÉFÉRENCES LÉGALES OBLIGATOIRES** :
   - TOUJOURS citer le numéro de l'article du CGI (ex: "Article 19 du CGI", "Article 73 du CGI")
   - TOUJOURS mentionner la loi de finances applicable si pertinent
   - TOUJOURS donner la date d'entrée en vigueur si disponible

2. **PRÉCISION DES TAUX ET CHIFFRES** :
   - Cite les taux EXACTS comme ils apparaissent dans le texte (ex: 20%, 31%, 37%)
   - Si un taux a des tranches ou seuils, mentionne-les précisément
   - **Liste TOUS les taux/tranches applicables** (même les cas spécifiques comme 40% ou 8,75%)
   - Ne jamais inventer ou approximer un chiffre

3. **TYPES D'IMPÔTS** :
   - IS (Impôt sur les Sociétés) : taux, base imposable, exonérations
   - IR (Impôt sur le Revenu) : barème, déductions, abattements
   - TVA : taux, exonérations, régimes spéciaux
   - Droits d'enregistrement et timbre

4. **FORMAT DE RÉPONSE** :
   - Commence par la référence légale complète (article du CGI)
   - Donne la réponse précise avec les chiffres exacts
   - **NE PAS citer l'historique législatif** (ex: "modifié par la LF 2023, puis LF 2024"). Donne directement le taux actuel appliqué.
   - Termine par "Source: [référence de l'article]"

5. **SI INFORMATION ABSENTE** :
   - Dis clairement : "Cette information n'est pas présente dans les documents fournis."
   - Ne JAMAIS inventer de taux ou d'article

CONTEXTE :
{context}

QUESTION : {question}

RÉPONSE (avec références légales complètes) :"""
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