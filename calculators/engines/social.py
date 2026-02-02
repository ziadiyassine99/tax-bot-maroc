"""
Social calculation engines - CNSS, AMO, indemnités, etc.
All calculations based on Moroccan social security regulations.
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with French formatting (comma decimal, space thousands)."""
    if value == int(value):
        return f"{int(value):,}".replace(",", " ")
    return f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",")


def calc_cotisations_cnss(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate CNSS social security contributions.
    
    Inputs:
        - salaire_brut: Gross monthly salary
        - taux_salarial: Employee rate (default 4.48%)
        - taux_patronal: Employer rate (default 8.98%)
    
    Returns structured result with calculations.
    """
    salaire = float(inputs.get('salaire_brut', 0))
    taux_sal = float(inputs.get('taux_salarial', 4.48))
    taux_pat = float(inputs.get('taux_patronal', 8.98))
    
    if salaire <= 0:
        return {
            'success': False,
            'error': 'Le salaire brut doit être supérieur à 0',
            'result': {},
            'table': [],
            'notes': []
        }
    
    # CNSS salary cap for certain contributions
    plafond_cnss = 6000
    salaire_plafonne = min(salaire, plafond_cnss)
    
    part_salariale = salaire_plafonne * taux_sal / 100
    part_patronale = salaire_plafonne * taux_pat / 100
    total_cotisation = part_salariale + part_patronale
    salaire_net = salaire - part_salariale
    
    return {
        'success': True,
        'result': {
            'salaire_brut': salaire,
            'salaire_plafonne': salaire_plafonne,
            'taux_salarial': taux_sal,
            'taux_patronal': taux_pat,
            'part_salariale': round(part_salariale, 2),
            'part_patronale': round(part_patronale, 2),
            'total_cotisation': round(total_cotisation, 2),
            'salaire_net': round(salaire_net, 2),
        },
        'table': [
            {'label': 'Salaire brut', 'value': f"{format_number(salaire)} DH"},
            {'label': 'Salaire plafonné (max 6000 DH)', 'value': f"{format_number(salaire_plafonne)} DH"},
            {'label': f'Part salariale ({taux_sal}%)', 'value': f"{format_number(part_salariale)} DH"},
            {'label': f'Part patronale ({taux_pat}%)', 'value': f"{format_number(part_patronale)} DH"},
            {'label': 'Total cotisations', 'value': f"{format_number(total_cotisation)} DH"},
            {'label': 'Salaire net (après cotisations)', 'value': f"{format_number(salaire_net)} DH"},
        ],
        'notes': [
            "Le salaire est plafonné à 6 000 DH pour le calcul des cotisations CNSS.",
            "Taux standards: 4,48% salarial + 8,98% patronal pour les prestations sociales."
        ]
    }


def calc_indemnite_maladie(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate daily sickness/accident allowance.
    
    Inputs:
        - salaire_moyen: Average salary of last 6 months
        - jours_arret: Number of sick days
    
    Rules:
        - Starts from 4th day of incapacity
        - 2/3 of daily salary
        - Max 52 weeks (364 days) over 24 months
        - Salary capped at 6000 DH
    """
    salaire = float(inputs.get('salaire_moyen', 0))
    jours = int(inputs.get('jours_arret', 0))
    
    if salaire <= 0:
        return {
            'success': False,
            'error': 'Le salaire moyen doit être supérieur à 0',
            'result': {},
            'table': [],
            'notes': []
        }
    
    if jours < 4:
        return {
            'success': True,
            'result': {'eligible': False, 'jours_arret': jours},
            'table': [
                {'label': 'Jours d\'arrêt', 'value': str(jours)},
                {'label': 'Éligibilité', 'value': 'Non éligible'},
            ],
            'notes': [
                "L'indemnité journalière n'est due qu'à partir du 4ème jour d'incapacité.",
                "Les 3 premiers jours constituent le délai de carence."
            ]
        }
    
    # Cap salary at 6000 DH
    salaire_plafonne = min(salaire, 6000)
    
    # Daily salary
    salaire_journalier = salaire_plafonne / 30
    
    # Allowance = 2/3 of daily salary
    indemnite_journaliere = salaire_journalier * (2/3)
    
    # Eligible days (excluding first 3 days)
    jours_indemnises = min(jours - 3, 364)  # Max 364 days
    
    # Total allowance
    montant_total = indemnite_journaliere * jours_indemnises
    
    return {
        'success': True,
        'result': {
            'eligible': True,
            'salaire_moyen': salaire,
            'salaire_plafonne': salaire_plafonne,
            'salaire_journalier': round(salaire_journalier, 2),
            'indemnite_journaliere': round(indemnite_journaliere, 2),
            'jours_arret': jours,
            'jours_indemnises': jours_indemnises,
            'montant_total': round(montant_total, 2),
        },
        'table': [
            {'label': 'Salaire moyen', 'value': f"{format_number(salaire)} DH"},
            {'label': 'Salaire plafonné', 'value': f"{format_number(salaire_plafonne)} DH"},
            {'label': 'Salaire journalier', 'value': f"{format_number(salaire_journalier)} DH"},
            {'label': 'Indemnité journalière (2/3)', 'value': f"{format_number(indemnite_journaliere)} DH"},
            {'label': 'Jours d\'arrêt', 'value': str(jours)},
            {'label': 'Jours indemnisés', 'value': str(jours_indemnises)},
            {'label': 'Montant total', 'value': f"{format_number(montant_total)} DH"},
        ],
        'notes': [
            "L'indemnité commence à partir du 4ème jour d'incapacité.",
            "Elle représente 2/3 du salaire journalier moyen plafonné.",
            "Durée maximale: 52 semaines (364 jours) sur 24 mois.",
            "Le salaire est plafonné à 6 000 DH."
        ]
    }


def calc_indemnite_maternite(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate maternity allowance.
    
    Inputs:
        - salaire_moyen: Average salary of last 6 months
    
    Rules:
        - 14 weeks (98 days) of leave
        - Daily salary capped at 6000/30 DH
    """
    salaire = float(inputs.get('salaire_moyen', 0))
    
    if salaire <= 0:
        return {
            'success': False,
            'error': 'Le salaire moyen doit être supérieur à 0',
            'result': {},
            'table': [],
            'notes': []
        }
    
    # Cap salary at 6000 DH
    salaire_plafonne = min(salaire, 6000)
    
    # Daily salary
    salaire_journalier = salaire_plafonne / 30
    
    # Maternity leave = 98 days (14 weeks)
    jours_maternite = 98
    montant_total = salaire_journalier * jours_maternite
    
    return {
        'success': True,
        'result': {
            'salaire_moyen': salaire,
            'salaire_plafonne': salaire_plafonne,
            'salaire_journalier': round(salaire_journalier, 2),
            'jours_maternite': jours_maternite,
            'montant_total': round(montant_total, 2),
        },
        'table': [
            {'label': 'Salaire moyen', 'value': f"{format_number(salaire)} DH"},
            {'label': 'Salaire plafonné', 'value': f"{format_number(salaire_plafonne)} DH"},
            {'label': 'Salaire journalier', 'value': f"{format_number(salaire_journalier)} DH"},
            {'label': 'Durée du congé', 'value': f"{jours_maternite} jours (14 semaines)"},
            {'label': 'Indemnité de maternité', 'value': f"{format_number(montant_total)} DH"},
        ],
        'notes': [
            "L'indemnité de maternité correspond à 14 semaines (98 jours).",
            "Le salaire est plafonné à 6 000 DH par mois pour le calcul."
        ]
    }


def calc_conge_naissance(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate birth leave allowance.
    
    Inputs:
        - salaire_mensuel: Monthly salary
    
    Rules:
        - 3 days of leave
        - Salary/26 per day
        - Capped at 692.30 DH total
    """
    salaire = float(inputs.get('salaire_mensuel', 0))
    
    if salaire <= 0:
        return {
            'success': False,
            'error': 'Le salaire mensuel doit être supérieur à 0',
            'result': {},
            'table': [],
            'notes': []
        }
    
    # Cap salary at 6000 DH
    salaire_plafonne = min(salaire, 6000)
    
    # Birth leave = 3 days × (salary/26)
    montant_calcule = salaire_plafonne * 3 / 26
    montant_max = 692.30
    montant_final = min(montant_calcule, montant_max)
    
    return {
        'success': True,
        'result': {
            'salaire_mensuel': salaire,
            'salaire_plafonne': salaire_plafonne,
            'montant_calcule': round(montant_calcule, 2),
            'montant_final': round(montant_final, 2),
            'plafonne': montant_calcule > montant_max,
        },
        'table': [
            {'label': 'Salaire mensuel', 'value': f"{format_number(salaire)} DH"},
            {'label': 'Durée du congé', 'value': '3 jours'},
            {'label': 'Montant du congé de naissance', 'value': f"{format_number(montant_final)} DH"},
        ],
        'notes': [
            "Le congé de naissance correspond à 3 jours de rémunération.",
            "Le montant est plafonné à 692,30 DH."
        ]
    }


def calc_ipd_perte_emploi(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate unemployment benefit (Indemnité pour Perte d'Emploi).
    
    Inputs:
        - salaire_net: Net monthly salary
    
    Rules:
        - 70% of net salary
        - Capped at 2570.86 DH
        - Duration: 6 months
    """
    salaire = float(inputs.get('salaire_net', 0))
    
    if salaire <= 0:
        return {
            'success': False,
            'error': 'Le salaire net doit être supérieur à 0',
            'result': {},
            'table': [],
            'notes': []
        }
    
    # IPD = 70% of net salary
    ipd_calcule = salaire * 70 / 100
    plafond = 2570.86
    ipd_final = min(ipd_calcule, plafond)
    
    return {
        'success': True,
        'result': {
            'salaire_net': salaire,
            'ipd_calcule': round(ipd_calcule, 2),
            'ipd_final': round(ipd_final, 2),
            'plafonne': ipd_calcule > plafond,
        },
        'table': [
            {'label': 'Salaire net mensuel', 'value': f"{format_number(salaire)} DH"},
            {'label': 'IPD (70% du salaire)', 'value': f"{format_number(ipd_calcule)} DH"},
            {'label': 'Indemnité mensuelle', 'value': f"{format_number(ipd_final)} DH"},
            {'label': 'Durée maximale', 'value': '6 mois'},
        ],
        'notes': [
            "L'IPD représente 70% du salaire net mensuel.",
            "Elle est plafonnée à 2 570,86 DH par mois.",
            "Durée: 6 mois maximum."
        ]
    }


def calc_pension_invalidite(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate invalidity pension.
    
    Inputs:
        - salaire_moyen: Average salary of last 6 months
        - jours_assurance: Number of insurance days
        - tierce_personne: Whether assistance is needed (0/1)
    
    Rules:
        - Minimum 1080 insurance days required
        - Base: 50% of capped salary
        - Majoration based on insurance days
        - +10% if third-party assistance needed
        - Min 1000 DH, Max 4200 DH
    """
    salaire = float(inputs.get('salaire_moyen', 0))
    jours = int(inputs.get('jours_assurance', 0))
    tierce = int(inputs.get('tierce_personne', 0))
    
    if jours < 1080:
        return {
            'success': True,
            'result': {'eligible': False, 'jours_assurance': jours},
            'table': [
                {'label': 'Jours d\'assurance', 'value': str(jours)},
                {'label': 'Éligibilité', 'value': 'Non éligible'},
            ],
            'notes': [
                "La pension d'invalidité nécessite au minimum 1080 jours d'assurance.",
                f"Il vous manque {1080 - jours} jours."
            ]
        }
    
    # Cap salary
    salaire_plafonne = min(salaire, 6000)
    
    # Calculate majoration based on insurance days
    major_pct = 0
    if jours >= 7560:
        major_pct = 20
    elif jours >= 7344:
        major_pct = 19
    elif jours >= 7128:
        major_pct = 18
    elif jours >= 6912:
        major_pct = 17
    elif jours >= 6696:
        major_pct = 16
    elif jours >= 6480:
        major_pct = 15
    elif jours >= 6264:
        major_pct = 14
    elif jours >= 6048:
        major_pct = 13
    elif jours >= 5832:
        major_pct = 12
    elif jours >= 5616:
        major_pct = 11
    elif jours >= 5400:
        major_pct = 10
    elif jours >= 5184:
        major_pct = 9
    elif jours >= 4968:
        major_pct = 8
    elif jours >= 4752:
        major_pct = 7
    elif jours >= 4536:
        major_pct = 6
    elif jours >= 4320:
        major_pct = 5
    elif jours >= 4104:
        major_pct = 4
    elif jours >= 3888:
        major_pct = 3
    elif jours >= 3672:
        major_pct = 2
    elif jours >= 3456:
        major_pct = 1
    
    # Calculate pension
    taux_base = 50
    taux_total = taux_base + major_pct
    if tierce == 1:
        taux_total += 10
    
    pension = salaire_plafonne * taux_total / 100
    
    # Apply min/max
    pension = max(1000, min(4200, pension))
    
    return {
        'success': True,
        'result': {
            'eligible': True,
            'salaire_plafonne': salaire_plafonne,
            'jours_assurance': jours,
            'taux_majoration': major_pct,
            'tierce_personne': tierce == 1,
            'pension_mensuelle': round(pension, 2),
        },
        'table': [
            {'label': 'Salaire plafonné', 'value': f"{format_number(salaire_plafonne)} DH"},
            {'label': 'Jours d\'assurance', 'value': str(jours)},
            {'label': 'Taux de base', 'value': '50%'},
            {'label': 'Majoration ancienneté', 'value': f"+{major_pct}%"},
            {'label': 'Assistance tierce personne', 'value': '+10%' if tierce == 1 else 'Non'},
            {'label': 'Pension mensuelle', 'value': f"{format_number(pension)} DH"},
        ],
        'notes': [
            "La pension est comprise entre 1 000 DH et 4 200 DH.",
            "Le salaire est plafonné à 6 000 DH.",
            "Majoration de 1% par tranche de 216 jours au-delà de 3456 jours."
        ]
    }
