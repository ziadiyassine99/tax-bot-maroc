"""
Calculator registry - Defines all available calculators with metadata.
"""

from typing import Dict, Any, List, Optional
from .engines import (
    # Social
    calc_cotisations_cnss,
    calc_indemnite_maladie,
    calc_indemnite_maternite,
    calc_conge_naissance,
    calc_ipd_perte_emploi,
    calc_pension_invalidite,
    # Fiscal
    calc_heures_supplementaires,
    calc_conges_payes,
    calc_amortissement_lineaire,
    calc_auto_entrepreneur,
    calc_sanctions_retard,
    # Ratios
    calc_liquidite_generale,
    calc_liquidite_reduite,
    calc_marge_nette,
    calc_roe,
    calc_endettement_global,
    calc_autonomie_financiere,
)


# Calculator definitions with metadata
CALCULATORS: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # SOCIAL / CNSS CALCULATORS
    # =========================================================================
    'cotisations_cnss': {
        'id': 'cotisations_cnss',
        'title': 'Cotisations CNSS',
        'description': 'Calculez vos cotisations salariales et patronales CNSS',
        'icon': '🏦',
        'category': 'social',
        'engine': calc_cotisations_cnss,
        'trigger_phrases': [
            'cotisation', 'cnss', 'cotisations sociales', 'part salariale',
            'part patronale', 'charges sociales', 'calculer cnss',
            'combien cotiser', 'montant cnss'
        ],
        'fields': [
            {'id': 'salaire_brut', 'label': 'Salaire brut mensuel (DH)', 'type': 'number', 'required': True, 'default': 10000},
            {'id': 'taux_salarial', 'label': 'Taux salarial (%)', 'type': 'number', 'required': False, 'default': 4.48},
            {'id': 'taux_patronal', 'label': 'Taux patronal (%)', 'type': 'number', 'required': False, 'default': 8.98},
        ],
    },
    
    'indemnite_maladie': {
        'id': 'indemnite_maladie',
        'title': 'Indemnité journalière de maladie',
        'description': 'Calculez votre indemnité en cas d\'arrêt maladie',
        'icon': '🏥',
        'category': 'social',
        'engine': calc_indemnite_maladie,
        'trigger_phrases': [
            'indemnité maladie', 'arrêt maladie', 'indemnité journalière',
            'maladie cnss', 'ijm', 'accident travail', 'jours maladie',
            'combien maladie', 'arrêt de travail'
        ],
        'fields': [
            {'id': 'salaire_moyen', 'label': 'Salaire moyen des 6 derniers mois (DH)', 'type': 'number', 'required': True, 'default': 6000},
            {'id': 'jours_arret', 'label': 'Nombre de jours d\'arrêt', 'type': 'number', 'required': True, 'default': 30},
        ],
    },
    
    'indemnite_maternite': {
        'id': 'indemnite_maternite',
        'title': 'Indemnité de maternité',
        'description': 'Calculez votre indemnité de congé maternité',
        'icon': '🤰',
        'category': 'social',
        'engine': calc_indemnite_maternite,
        'trigger_phrases': [
            'maternité', 'congé maternité', 'indemnité maternité',
            'grossesse', 'accouchement', 'enceinte', 'naissance bébé',
            'allocations maternité'
        ],
        'fields': [
            {'id': 'salaire_moyen', 'label': 'Salaire moyen des 6 derniers mois (DH)', 'type': 'number', 'required': True, 'default': 6000},
        ],
    },
    
    'conge_naissance': {
        'id': 'conge_naissance',
        'title': 'Congé de naissance',
        'description': 'Calculez l\'indemnité du congé de naissance (père)',
        'icon': '👶',
        'category': 'social',
        'engine': calc_conge_naissance,
        'trigger_phrases': [
            'congé naissance', 'naissance enfant', 'congé paternité',
            'père naissance', '3 jours naissance', 'indemnité naissance'
        ],
        'fields': [
            {'id': 'salaire_mensuel', 'label': 'Salaire mensuel (DH)', 'type': 'number', 'required': True, 'default': 6000},
        ],
    },
    
    'ipd_perte_emploi': {
        'id': 'ipd_perte_emploi',
        'title': 'Indemnité perte d\'emploi (IPD)',
        'description': 'Calculez votre indemnité chômage IPD',
        'icon': '💼',
        'category': 'social',
        'engine': calc_ipd_perte_emploi,
        'trigger_phrases': [
            'perte emploi', 'chômage', 'ipd', 'licenciement',
            'indemnité chômage', 'perte de travail', 'sans emploi',
            'allocation chômage'
        ],
        'fields': [
            {'id': 'salaire_net', 'label': 'Salaire net mensuel (DH)', 'type': 'number', 'required': True, 'default': 8000},
        ],
    },
    
    'pension_invalidite': {
        'id': 'pension_invalidite',
        'title': 'Pension d\'invalidité',
        'description': 'Calculez votre pension d\'invalidité CNSS',
        'icon': '♿',
        'category': 'social',
        'engine': calc_pension_invalidite,
        'trigger_phrases': [
            'pension invalidité', 'invalidité', 'handicap',
            'incapacité permanente', 'pension cnss', 'invalide'
        ],
        'fields': [
            {'id': 'salaire_moyen', 'label': 'Salaire moyen des 6 derniers mois (DH)', 'type': 'number', 'required': True, 'default': 6000},
            {'id': 'jours_assurance', 'label': 'Jours d\'assurance cumulés', 'type': 'number', 'required': True, 'default': 3000},
            {'id': 'tierce_personne', 'label': 'Assistance tierce personne', 'type': 'select', 'required': False, 'default': 0, 'options': [{'value': 0, 'label': 'Non'}, {'value': 1, 'label': 'Oui'}]},
        ],
    },
    
    # =========================================================================
    # FISCAL CALCULATORS
    # =========================================================================
    'heures_supplementaires': {
        'id': 'heures_supplementaires',
        'title': 'Heures supplémentaires',
        'description': 'Calculez la rémunération de vos heures supplémentaires',
        'icon': '⏰',
        'category': 'fiscal',
        'engine': calc_heures_supplementaires,
        'trigger_phrases': [
            'heures supplémentaires', 'heures sup', 'overtime',
            'majoration heures', 'travail supplémentaire', '25%', '50%', '100%',
            'rémunération heures'
        ],
        'fields': [
            {'id': 'salaire_mensuel', 'label': 'Salaire mensuel brut (DH)', 'type': 'number', 'required': True, 'default': 5000},
            {'id': 'heures_25', 'label': 'Heures à 25% (jour 6h-21h)', 'type': 'number', 'required': False, 'default': 0},
            {'id': 'heures_50_nuit', 'label': 'Heures à 50% (nuit 21h-6h)', 'type': 'number', 'required': False, 'default': 0},
            {'id': 'heures_50_repos', 'label': 'Heures à 50% (repos/férié jour)', 'type': 'number', 'required': False, 'default': 0},
            {'id': 'heures_100', 'label': 'Heures à 100% (repos/férié nuit)', 'type': 'number', 'required': False, 'default': 0},
        ],
    },
    
    'conges_payes': {
        'id': 'conges_payes',
        'title': 'Congés payés',
        'description': 'Calculez vos droits aux congés payés',
        'icon': '🏖️',
        'category': 'fiscal',
        'engine': calc_conges_payes,
        'trigger_phrases': [
            'congés payés', 'jours de congé', 'vacances',
            'droit aux congés', 'calcul congés', 'combien de jours',
            'congé annuel'
        ],
        'fields': [
            {'id': 'date_naissance', 'label': 'Date de naissance (JJ/MM/AAAA)', 'type': 'text', 'required': True, 'default': '01/01/1990'},
            {'id': 'date_embauche', 'label': 'Date d\'embauche (JJ/MM/AAAA)', 'type': 'text', 'required': True, 'default': '01/01/2020'},
            {'id': 'fonction', 'label': 'Fonction', 'type': 'select', 'required': False, 'default': 1, 'options': [{'value': 1, 'label': 'Employé'}, {'value': 2, 'label': 'Cadre'}]},
        ],
    },
    
    'amortissement': {
        'id': 'amortissement',
        'title': 'Amortissement linéaire',
        'description': 'Calculez le tableau d\'amortissement d\'une immobilisation',
        'icon': '📊',
        'category': 'fiscal',
        'engine': calc_amortissement_lineaire,
        'trigger_phrases': [
            'amortissement', 'dotation aux amortissements', 'immobilisation',
            'dépréciation', 'tableau amortissement', 'amortir',
            'durée amortissement', 'taux amortissement'
        ],
        'fields': [
            {'id': 'type_bien', 'label': 'Type de bien', 'type': 'select', 'required': True, 'default': 3, 'options': [
                {'value': 1, 'label': 'Constructions (25 ans)'},
                {'value': 2, 'label': 'Agencements (20 ans)'},
                {'value': 3, 'label': 'Matériel et outillage (10 ans)'},
                {'value': 4, 'label': 'Mobilier de bureau (10 ans)'},
                {'value': 5, 'label': 'Matériel informatique (10 ans)'},
                {'value': 6, 'label': 'Matériel de transport (5 ans)'},
                {'value': 7, 'label': 'Véhicules (5 ans)'},
                {'value': 8, 'label': 'Voiture de tourisme (5 ans)'},
                {'value': 9, 'label': 'Logiciels (3-4 ans)'},
            ]},
            {'id': 'prix_acquisition', 'label': 'Prix d\'acquisition (DH)', 'type': 'number', 'required': True, 'default': 100000},
            {'id': 'mois_acquisition', 'label': 'Mois d\'acquisition (1-12)', 'type': 'number', 'required': True, 'default': 1},
            {'id': 'annee_acquisition', 'label': 'Année d\'acquisition', 'type': 'number', 'required': True, 'default': 2024},
        ],
    },
    
    'auto_entrepreneur': {
        'id': 'auto_entrepreneur',
        'title': 'Impôt Auto-entrepreneur',
        'description': 'Calculez votre impôt en tant qu\'auto-entrepreneur',
        'icon': '🧾',
        'category': 'fiscal',
        'engine': calc_auto_entrepreneur,
        'trigger_phrases': [
            'auto-entrepreneur', 'auto entrepreneur', 'autoentrepreneur',
            'impôt ae', 'statut ae', 'régime ae', 'taxe auto-entrepreneur',
            'contribution ae'
        ],
        'fields': [
            {'id': 'chiffre_affaires', 'label': 'Chiffre d\'affaires annuel (DH)', 'type': 'number', 'required': True, 'default': 100000},
            {'id': 'type_activite', 'label': 'Type d\'activité', 'type': 'select', 'required': True, 'default': 1, 'options': [
                {'value': 1, 'label': 'Commerciale/Industrielle (0.5%)'},
                {'value': 2, 'label': 'Artisanale (1%)'},
                {'value': 3, 'label': 'Services (2%)'},
            ]},
        ],
    },
    
    'sanctions_retard': {
        'id': 'sanctions_retard',
        'title': 'Sanctions pour retard',
        'description': 'Calculez les majorations pour retard de paiement/dépôt',
        'icon': '⚠️',
        'category': 'fiscal',
        'engine': calc_sanctions_retard,
        'trigger_phrases': [
            'retard', 'majoration', 'pénalité', 'sanction fiscale',
            'retard paiement', 'retard dépôt', 'amende fiscale',
            'intérêts de retard'
        ],
        'fields': [
            {'id': 'montant', 'label': 'Montant de l\'impôt dû (DH)', 'type': 'number', 'required': True, 'default': 10000},
            {'id': 'mois_retard', 'label': 'Mois de retard', 'type': 'number', 'required': True, 'default': 3},
            {'id': 'type_retard', 'label': 'Type de retard', 'type': 'select', 'required': True, 'default': 2, 'options': [
                {'value': 1, 'label': 'Dépôt tardif'},
                {'value': 2, 'label': 'Paiement tardif'},
            ]},
        ],
    },
    
    # =========================================================================
    # RATIO CALCULATORS
    # =========================================================================
    'liquidite_generale': {
        'id': 'liquidite_generale',
        'title': 'Ratio de liquidité générale',
        'description': 'Évaluez la capacité à couvrir les dettes à court terme',
        'icon': '💧',
        'category': 'ratios',
        'engine': calc_liquidite_generale,
        'trigger_phrases': [
            'liquidité générale', 'current ratio', 'ratio liquidité',
            'solvabilité court terme', 'actif circulant', 'passif circulant'
        ],
        'fields': [
            {'id': 'actif_circulant', 'label': 'Actif circulant (DH)', 'type': 'number', 'required': True, 'default': 500000},
            {'id': 'passif_circulant', 'label': 'Passif circulant (DH)', 'type': 'number', 'required': True, 'default': 300000},
        ],
    },
    
    'liquidite_reduite': {
        'id': 'liquidite_reduite',
        'title': 'Ratio de liquidité réduite',
        'description': 'Quick ratio - liquidité hors stocks',
        'icon': '💦',
        'category': 'ratios',
        'engine': calc_liquidite_reduite,
        'trigger_phrases': [
            'liquidité réduite', 'quick ratio', 'acid test',
            'liquidité hors stock', 'ratio acide'
        ],
        'fields': [
            {'id': 'actif_circulant', 'label': 'Actif circulant (DH)', 'type': 'number', 'required': True, 'default': 500000},
            {'id': 'stocks', 'label': 'Stocks (DH)', 'type': 'number', 'required': True, 'default': 150000},
            {'id': 'passif_circulant', 'label': 'Passif circulant (DH)', 'type': 'number', 'required': True, 'default': 300000},
        ],
    },
    
    'marge_nette': {
        'id': 'marge_nette',
        'title': 'Marge nette',
        'description': 'Calculez la rentabilité nette sur le chiffre d\'affaires',
        'icon': '📈',
        'category': 'ratios',
        'engine': calc_marge_nette,
        'trigger_phrases': [
            'marge nette', 'rentabilité nette', 'profit margin',
            'bénéfice net', 'résultat net', 'marge bénéficiaire'
        ],
        'fields': [
            {'id': 'resultat_net', 'label': 'Résultat net (DH)', 'type': 'number', 'required': True, 'default': 100000},
            {'id': 'chiffre_affaires', 'label': 'Chiffre d\'affaires (DH)', 'type': 'number', 'required': True, 'default': 1000000},
        ],
    },
    
    'roe': {
        'id': 'roe',
        'title': 'Rentabilité des fonds propres (ROE)',
        'description': 'Return on Equity - rentabilité pour les actionnaires',
        'icon': '🎯',
        'category': 'ratios',
        'engine': calc_roe,
        'trigger_phrases': [
            'roe', 'return on equity', 'rentabilité fonds propres',
            'rentabilité capitaux', 'rendement actionnaires'
        ],
        'fields': [
            {'id': 'resultat_net', 'label': 'Résultat net (DH)', 'type': 'number', 'required': True, 'default': 100000},
            {'id': 'capitaux_propres', 'label': 'Capitaux propres (DH)', 'type': 'number', 'required': True, 'default': 500000},
        ],
    },
    
    'endettement': {
        'id': 'endettement',
        'title': 'Ratio d\'endettement global',
        'description': 'Mesurez le niveau d\'endettement de l\'entreprise',
        'icon': '🏦',
        'category': 'ratios',
        'engine': calc_endettement_global,
        'trigger_phrases': [
            'endettement', 'ratio dette', 'niveau dettes',
            'structure financière', 'dettes entreprise', 'leverage'
        ],
        'fields': [
            {'id': 'total_dettes', 'label': 'Total dettes (DH)', 'type': 'number', 'required': True, 'default': 400000},
            {'id': 'total_passif', 'label': 'Total passif / bilan (DH)', 'type': 'number', 'required': True, 'default': 1000000},
        ],
    },
    
    'autonomie_financiere': {
        'id': 'autonomie_financiere',
        'title': 'Autonomie financière',
        'description': 'Évaluez l\'indépendance financière de l\'entreprise',
        'icon': '🏛️',
        'category': 'ratios',
        'engine': calc_autonomie_financiere,
        'trigger_phrases': [
            'autonomie financière', 'indépendance financière',
            'capitaux propres dettes', 'structure capital'
        ],
        'fields': [
            {'id': 'capitaux_propres', 'label': 'Capitaux propres (DH)', 'type': 'number', 'required': True, 'default': 600000},
            {'id': 'dettes_financement', 'label': 'Dettes de financement (DH)', 'type': 'number', 'required': True, 'default': 400000},
        ],
    },
}


def get_calculator(calculator_id: str) -> Optional[Dict[str, Any]]:
    """Get a calculator by its ID."""
    return CALCULATORS.get(calculator_id)


def get_calculators_by_category(category: str) -> List[Dict[str, Any]]:
    """Get all calculators in a category."""
    return [calc for calc in CALCULATORS.values() if calc['category'] == category]


def get_all_calculators() -> List[Dict[str, Any]]:
    """Get all calculators."""
    return list(CALCULATORS.values())


def execute_calculator(calculator_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a calculator with given inputs."""
    calculator = get_calculator(calculator_id)
    if not calculator:
        return {
            'success': False,
            'error': f'Calculateur non trouvé: {calculator_id}',
            'result': {},
            'table': [],
            'notes': []
        }
    
    engine = calculator['engine']
    return engine(inputs)
