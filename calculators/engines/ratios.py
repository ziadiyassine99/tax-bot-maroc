"""
Financial ratio calculation engines.
All calculations based on standard financial analysis practices.
"""

from typing import Dict, Any


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with French formatting."""
    if value == int(value):
        return f"{int(value):,}".replace(",", " ")
    return f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",")


def calc_liquidite_generale(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate general liquidity ratio (Current Ratio).
    
    Formula: Current Assets / Current Liabilities
    
    Inputs:
        - actif_circulant: Current assets
        - passif_circulant: Current liabilities
    """
    actif = float(inputs.get('actif_circulant', 0))
    passif = float(inputs.get('passif_circulant', 0))
    
    if passif == 0:
        return {
            'success': False,
            'error': 'Le passif circulant ne peut pas être égal à zéro',
            'result': {},
            'table': [],
            'notes': []
        }
    
    ratio = actif / passif
    
    if ratio < 1:
        interpretation = "Ratio inférieur à 1: L'actif circulant ne couvre pas les dettes à court terme. Situation préoccupante."
    elif ratio == 1:
        interpretation = "Ratio égal à 1: Couverture juste des dettes. Situation fragile."
    elif ratio < 1.5:
        interpretation = "Ratio acceptable mais à surveiller."
    else:
        interpretation = "Bon ratio de liquidité. L'entreprise est solvable à court terme."
    
    return {
        'success': True,
        'result': {
            'actif_circulant': actif,
            'passif_circulant': passif,
            'ratio': round(ratio, 4),
            'solvable': ratio >= 1,
        },
        'table': [
            {'label': 'Actif circulant', 'value': f"{format_number(actif)} DH"},
            {'label': 'Passif circulant', 'value': f"{format_number(passif)} DH"},
            {'label': 'Ratio de liquidité générale', 'value': f"{format_number(ratio, 2)}"},
            {'label': 'Interprétation', 'value': interpretation},
        ],
        'notes': [
            "Un ratio supérieur à 1 indique que l'entreprise peut couvrir ses dettes à court terme.",
            "Un ratio de 1,5 à 2 est généralement considéré comme sain."
        ]
    }


def calc_liquidite_reduite(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate quick ratio (Acid Test).
    
    Formula: (Current Assets - Inventory) / Current Liabilities
    
    Inputs:
        - actif_circulant: Current assets
        - stocks: Inventory
        - passif_circulant: Current liabilities
    """
    actif = float(inputs.get('actif_circulant', 0))
    stocks = float(inputs.get('stocks', 0))
    passif = float(inputs.get('passif_circulant', 0))
    
    if passif == 0:
        return {
            'success': False,
            'error': 'Le passif circulant ne peut pas être égal à zéro',
            'result': {},
            'table': [],
            'notes': []
        }
    
    actif_hors_stocks = actif - stocks
    ratio = actif_hors_stocks / passif
    
    if ratio >= 1:
        interpretation = "Ratio satisfaisant. L'entreprise peut couvrir ses dettes sans vendre ses stocks."
    elif ratio >= 0.8:
        interpretation = "Ratio acceptable, mais à surveiller."
    else:
        interpretation = "Ratio insuffisant. Difficultés potentielles de trésorerie."
    
    return {
        'success': True,
        'result': {
            'actif_circulant': actif,
            'stocks': stocks,
            'actif_hors_stocks': actif_hors_stocks,
            'passif_circulant': passif,
            'ratio': round(ratio, 4),
        },
        'table': [
            {'label': 'Actif circulant', 'value': f"{format_number(actif)} DH"},
            {'label': 'Stocks', 'value': f"{format_number(stocks)} DH"},
            {'label': 'Actif hors stocks', 'value': f"{format_number(actif_hors_stocks)} DH"},
            {'label': 'Passif circulant', 'value': f"{format_number(passif)} DH"},
            {'label': 'Liquidité réduite', 'value': f"{format_number(ratio, 2)}"},
            {'label': 'Interprétation', 'value': interpretation},
        ],
        'notes': [
            "Ce ratio exclut les stocks car ils sont moins liquides.",
            "Un ratio ≥ 1 est généralement considéré comme satisfaisant."
        ]
    }


def calc_marge_nette(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate net profit margin.
    
    Formula: (Net Profit / Revenue) × 100
    
    Inputs:
        - resultat_net: Net profit
        - chiffre_affaires: Revenue
    """
    resultat = float(inputs.get('resultat_net', 0))
    ca = float(inputs.get('chiffre_affaires', 0))
    
    if ca == 0:
        return {
            'success': False,
            'error': 'Le chiffre d\'affaires ne peut pas être égal à zéro',
            'result': {},
            'table': [],
            'notes': []
        }
    
    marge = (resultat / ca) * 100
    
    if marge > 10:
        interpretation = "Excellente marge nette. Entreprise très rentable."
    elif marge > 5:
        interpretation = "Bonne marge nette. Rentabilité satisfaisante."
    elif marge > 2:
        interpretation = "Marge correcte mais à améliorer."
    elif marge > 0:
        interpretation = "Marge faible. Peu de bénéfice dégagé."
    else:
        interpretation = "Marge négative. L'entreprise est déficitaire."
    
    return {
        'success': True,
        'result': {
            'resultat_net': resultat,
            'chiffre_affaires': ca,
            'marge_nette': round(marge, 2),
        },
        'table': [
            {'label': 'Résultat net', 'value': f"{format_number(resultat)} DH"},
            {'label': 'Chiffre d\'affaires', 'value': f"{format_number(ca)} DH"},
            {'label': 'Marge nette', 'value': f"{format_number(marge)}%"},
            {'label': 'Interprétation', 'value': interpretation},
        ],
        'notes': [
            "La marge nette mesure le pourcentage de bénéfice sur chaque dirham de CA.",
            "Elle varie selon les secteurs d'activité."
        ]
    }


def calc_roe(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate Return on Equity (ROE).
    
    Formula: (Net Profit / Shareholders' Equity) × 100
    
    Inputs:
        - resultat_net: Net profit
        - capitaux_propres: Shareholders' equity
    """
    resultat = float(inputs.get('resultat_net', 0))
    capitaux = float(inputs.get('capitaux_propres', 0))
    
    if capitaux == 0:
        return {
            'success': False,
            'error': 'Les capitaux propres ne peuvent pas être égaux à zéro',
            'result': {},
            'table': [],
            'notes': []
        }
    
    roe = (resultat / capitaux) * 100
    
    if roe > 15:
        interpretation = "Excellent ROE. Très bonne rentabilité pour les actionnaires."
    elif roe > 10:
        interpretation = "Bon ROE. Performance satisfaisante."
    elif roe > 5:
        interpretation = "ROE moyen."
    elif roe > 0:
        interpretation = "ROE faible."
    else:
        interpretation = "ROE négatif. L'entreprise génère des pertes."
    
    return {
        'success': True,
        'result': {
            'resultat_net': resultat,
            'capitaux_propres': capitaux,
            'roe': round(roe, 2),
        },
        'table': [
            {'label': 'Résultat net', 'value': f"{format_number(resultat)} DH"},
            {'label': 'Capitaux propres', 'value': f"{format_number(capitaux)} DH"},
            {'label': 'Rentabilité des fonds propres (ROE)', 'value': f"{format_number(roe)}%"},
            {'label': 'Interprétation', 'value': interpretation},
        ],
        'notes': [
            "Le ROE mesure la rentabilité des capitaux investis par les actionnaires.",
            "Un ROE > 15% est généralement considéré comme excellent."
        ]
    }


def calc_endettement_global(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate global debt ratio.
    
    Formula: (Total Debt / Total Assets) × 100
    
    Inputs:
        - total_dettes: Total liabilities
        - total_passif: Total liabilities + equity (or total assets)
    """
    dettes = float(inputs.get('total_dettes', 0))
    passif = float(inputs.get('total_passif', 0))
    
    if passif == 0:
        return {
            'success': False,
            'error': 'Le total passif ne peut pas être égal à zéro',
            'result': {},
            'table': [],
            'notes': []
        }
    
    ratio = (dettes / passif) * 100
    
    if ratio < 50:
        interpretation = "Endettement faible. Bonne structure financière."
    elif ratio < 70:
        interpretation = "Endettement modéré. À surveiller."
    else:
        interpretation = "Endettement élevé. Forte dépendance aux créanciers."
    
    return {
        'success': True,
        'result': {
            'total_dettes': dettes,
            'total_passif': passif,
            'ratio_endettement': round(ratio, 2),
        },
        'table': [
            {'label': 'Total dettes', 'value': f"{format_number(dettes)} DH"},
            {'label': 'Total passif', 'value': f"{format_number(passif)} DH"},
            {'label': 'Ratio d\'endettement global', 'value': f"{format_number(ratio)}%"},
            {'label': 'Interprétation', 'value': interpretation},
        ],
        'notes': [
            "Ce ratio mesure la part des dettes dans le financement total.",
            "Un ratio < 50% est généralement considéré comme sain."
        ]
    }


def calc_autonomie_financiere(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate financial autonomy ratio.
    
    Formula: Shareholders' Equity / (Shareholders' Equity + Long-term Debt)
    
    Inputs:
        - capitaux_propres: Shareholders' equity
        - dettes_financement: Long-term financing debt
    """
    capitaux = float(inputs.get('capitaux_propres', 0))
    dettes = float(inputs.get('dettes_financement', 0))
    
    total = capitaux + dettes
    if total == 0:
        return {
            'success': False,
            'error': 'La somme des capitaux propres et dettes ne peut pas être nulle',
            'result': {},
            'table': [],
            'notes': []
        }
    
    # Ratio autonomie = Capitaux propres / Capitaux permanents
    ratio = capitaux / total
    
    # Alternative: Dettes / Capitaux propres
    ratio_dette_cp = dettes / capitaux if capitaux > 0 else float('inf')
    
    if ratio >= 0.5:
        interpretation = "Bonne autonomie financière. L'entreprise est majoritairement financée par ses fonds propres."
    elif ratio >= 0.33:
        interpretation = "Autonomie financière moyenne."
    else:
        interpretation = "Faible autonomie. Forte dépendance aux financements externes."
    
    return {
        'success': True,
        'result': {
            'capitaux_propres': capitaux,
            'dettes_financement': dettes,
            'ratio_autonomie': round(ratio, 4),
            'ratio_dette_cp': round(ratio_dette_cp, 4) if ratio_dette_cp != float('inf') else None,
        },
        'table': [
            {'label': 'Capitaux propres', 'value': f"{format_number(capitaux)} DH"},
            {'label': 'Dettes de financement', 'value': f"{format_number(dettes)} DH"},
            {'label': 'Ratio d\'autonomie financière', 'value': f"{format_number(ratio * 100)}%"},
            {'label': 'Interprétation', 'value': interpretation},
        ],
        'notes': [
            "Ce ratio mesure l'indépendance financière de l'entreprise.",
            "Un ratio > 50% indique une bonne autonomie."
        ]
    }
