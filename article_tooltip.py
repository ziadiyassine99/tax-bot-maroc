"""
Article Tooltip module - Makes article citations clickable with tooltips.
Extracts article numbers from responses and fetches full article content.
"""

import re
from typing import List, Dict, Set
import html


# Pattern to match article citations like "article 92", "l'article 101", "Article 8-1"
ARTICLE_PATTERN = r"(?:l['']\s*)?article\s+(\d+(?:[-.]\d+)?)"


def extract_article_citations(text: str) -> List[str]:
    """
    Extract unique article numbers from text.
    
    Args:
        text: The response text containing article citations
        
    Returns:
        List of unique article numbers (e.g., ["92", "101", "8-1"])
    """
    matches = re.findall(ARTICLE_PATTERN, text, re.IGNORECASE)
    # Return unique articles while preserving order
    seen: Set[str] = set()
    unique_articles = []
    for article in matches:
        if article not in seen:
            seen.add(article)
            unique_articles.append(article)
    return unique_articles


def fetch_article_content(rag_chain, article_num: str, module_name: str) -> str:
    """
    Fetch the full content of an article by querying the RAG chain.
    
    Args:
        rag_chain: The RAGChainBuilder instance
        article_num: The article number (e.g., "92")
        module_name: The module name for context (e.g., "CGI", "Code du Travail")
        
    Returns:
        The full article content as returned by the RAG
    """
    # Build internal query to fetch the complete article
    query = f"Cite moi l'article {article_num} complet du {module_name}. Donne uniquement le texte de l'article sans commentaire."
    
    try:
        # Use invoke to get the response (not streaming)
        response = rag_chain.invoke(query)
        return response
    except Exception as e:
        return f"Erreur lors de la récupération de l'article {article_num}: {str(e)}"


def format_response_with_tooltips(response: str, articles_content: Dict[str, str]) -> str:
    """
    Replace article citations in the response with clickable HTML elements
    that show tooltips with full article content.
    
    Args:
        response: The original response text
        articles_content: Dict mapping article numbers to their full content
        
    Returns:
        HTML-formatted response with clickable article tooltips
    """
    if not articles_content:
        return response
    
    def replace_article(match):
        """Replace an article citation with HTML tooltip."""
        full_match = match.group(0)  # e.g., "l'article 92" or "article 101"
        article_num = match.group(1)  # e.g., "92" or "101"
        
        if article_num not in articles_content:
            return full_match
        
        # Escape HTML in the article content for safety
        content = html.escape(articles_content[article_num])
        # Convert newlines to <br> for HTML display
        content = content.replace('\n', '<br>')
        
        # Create the tooltip HTML
        tooltip_html = f'''<span class="article-link" tabindex="0">
{full_match}
<span class="article-tooltip">
<strong>Article {article_num}</strong><br><br>
{content}
</span>
</span>'''
        
        return tooltip_html
    
    # Replace all article citations with tooltip HTML
    formatted = re.sub(ARTICLE_PATTERN, replace_article, response, flags=re.IGNORECASE)
    
    return formatted


def process_response_with_tooltips(response: str, rag_chain, module_name: str, 
                                    cached_articles: Dict[str, str] = None) -> tuple:
    """
    Complete pipeline: extract articles, fetch content, format response.
    
    Args:
        response: The original response text
        rag_chain: The RAGChainBuilder instance for fetching articles
        module_name: The module name (e.g., "CGI")
        cached_articles: Optional dict of already fetched articles to avoid re-fetching
        
    Returns:
        Tuple of (formatted_response, articles_content_dict)
    """
    if cached_articles is None:
        cached_articles = {}
    
    # Extract article numbers
    article_nums = extract_article_citations(response)
    
    if not article_nums:
        return response, cached_articles
    
    # Fetch content for articles not yet cached
    articles_to_fetch = [num for num in article_nums if num not in cached_articles]
    
    for article_num in articles_to_fetch:
        content = fetch_article_content(rag_chain, article_num, module_name)
        cached_articles[article_num] = content
    
    # Format response with tooltips
    formatted_response = format_response_with_tooltips(response, cached_articles)
    
    return formatted_response, cached_articles
