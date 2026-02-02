"""
Calculator matcher - Detects when to show calculators based on user input.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from .registry import CALCULATORS, get_calculator


# Keywords that indicate calculation intent
CALCULATION_KEYWORDS = [
    'calculer', 'calcul', 'combien', 'quel est le montant',
    'montant de', 'comment calculer', 'je veux calculer',
    'peux-tu calculer', 'aide-moi à calculer', 'simuler',
    'simulation', 'estimer', 'estimation', 'évaluer'
]

# Keywords that indicate user wants information, not calculation
INFO_KEYWORDS = [
    'qu\'est-ce que', 'c\'est quoi', 'définition',
    'expliquer', 'explication', 'comment fonctionne',
    'quel est le taux', 'quels sont les taux', 'règles',
    'conditions', 'qui peut', 'qui a droit', 'éligibilité'
]


def normalize_text(text: str) -> str:
    """Normalize text for matching (lowercase, remove accents, etc.)."""
    text = text.lower()
    # Simple accent removal
    replacements = {
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'à': 'a', 'â': 'a', 'ä': 'a',
        'î': 'i', 'ï': 'i',
        'ô': 'o', 'ö': 'o',
        'ù': 'u', 'û': 'u', 'ü': 'u',
        'ç': 'c', 'œ': 'oe', 'æ': 'ae'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def has_calculation_intent(query: str) -> bool:
    """Check if the query indicates calculation intent."""
    query_lower = normalize_text(query)
    
    # Check for calculation keywords
    for keyword in CALCULATION_KEYWORDS:
        if normalize_text(keyword) in query_lower:
            return True
    
    # Check for question patterns about amounts
    amount_patterns = [
        r'combien.*(?:cotis|payer|toucher|recevoir|coût)',
        r'quel.*montant',
        r'quelle.*somme',
        r'calculer',
        r'\d+.*(?:dh|dirhams?)',  # Contains numbers with currency
    ]
    
    for pattern in amount_patterns:
        if re.search(pattern, query_lower):
            return True
    
    return False


def is_info_query(query: str) -> bool:
    """Check if the query is asking for information rather than calculation."""
    query_lower = normalize_text(query)
    
    for keyword in INFO_KEYWORDS:
        if normalize_text(keyword) in query_lower:
            return True
    
    return False


def calculate_match_score(query: str, calculator: Dict[str, Any]) -> float:
    """
    Calculate a match score between query and calculator.
    Returns a score between 0 and 1.
    """
    query_normalized = normalize_text(query)
    score = 0.0
    max_possible = 0.0
    
    # Check trigger phrases (high weight)
    trigger_matches = 0
    for phrase in calculator.get('trigger_phrases', []):
        max_possible += 1.0
        phrase_normalized = normalize_text(phrase)
        if phrase_normalized in query_normalized:
            trigger_matches += 1
            score += 1.0
        elif any(word in query_normalized for word in phrase_normalized.split()):
            score += 0.3
    
    # Bonus for multiple trigger matches
    if trigger_matches > 1:
        score += 0.5
    
    # Check title match (medium weight)
    max_possible += 0.5
    title_normalized = normalize_text(calculator.get('title', ''))
    if title_normalized in query_normalized:
        score += 0.5
    elif any(word in query_normalized for word in title_normalized.split() if len(word) > 3):
        score += 0.2
    
    # Check description match (low weight)
    max_possible += 0.3
    desc_normalized = normalize_text(calculator.get('description', ''))
    if any(word in query_normalized for word in desc_normalized.split() if len(word) > 4):
        score += 0.1
    
    # Normalize score
    if max_possible > 0:
        return min(score / max_possible, 1.0)
    return 0.0


def find_matching_calculators(query: str, top_k: int = 3, threshold: float = 0.15) -> List[Tuple[str, float]]:
    """
    Find calculators that match the query.
    
    Args:
        query: User's question
        top_k: Maximum number of calculators to return
        threshold: Minimum score to consider a match
    
    Returns:
        List of (calculator_id, score) tuples, sorted by score descending
    """
    matches = []
    
    for calc_id, calculator in CALCULATORS.items():
        score = calculate_match_score(query, calculator)
        if score >= threshold:
            matches.append((calc_id, score))
    
    # Sort by score descending
    matches.sort(key=lambda x: x[1], reverse=True)
    
    return matches[:top_k]


def should_show_calculator(query: str, min_score: float = 0.2) -> Tuple[bool, Optional[str], float]:
    """
    Determine if a calculator should be shown for this query.
    
    Args:
        query: User's question
        min_score: Minimum match score to show calculator
    
    Returns:
        Tuple of (should_show, calculator_id, score)
    """
    # If user is clearly asking for info/explanation, don't show calculator
    if is_info_query(query) and not has_calculation_intent(query):
        return (False, None, 0.0)
    
    # Find best matching calculator
    matches = find_matching_calculators(query, top_k=1, threshold=min_score)
    
    if not matches:
        return (False, None, 0.0)
    
    calc_id, score = matches[0]
    
    # Higher threshold if no explicit calculation intent
    required_score = min_score if has_calculation_intent(query) else min_score + 0.1
    
    if score >= required_score:
        return (True, calc_id, score)
    
    return (False, None, 0.0)


def get_suggested_calculators(query: str, response: str) -> List[Dict[str, Any]]:
    """
    Get calculators to suggest after providing a RAG response.
    These are secondary suggestions based on the topic discussed.
    
    Args:
        query: Original user question
        response: RAG response that was given
    
    Returns:
        List of calculator info dicts to suggest
    """
    combined_text = f"{query} {response}"
    matches = find_matching_calculators(combined_text, top_k=2, threshold=0.1)
    
    suggestions = []
    for calc_id, score in matches:
        # Only suggest if score is moderate (not too high, not too low)
        # High scores should have already triggered the main calculator
        if 0.1 <= score < 0.4:
            calculator = get_calculator(calc_id)
            if calculator:
                suggestions.append({
                    'id': calc_id,
                    'title': calculator['title'],
                    'description': calculator['description'],
                    'icon': calculator['icon'],
                    'score': score
                })
    
    return suggestions
