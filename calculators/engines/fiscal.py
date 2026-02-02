"""
Fiscal calculation engines - IR, IS, TVA, amortissement, heures sup, etc.
All calculations based on Moroccan tax regulations (CGI).
"""

from typing import Dict, Any, List
from datetime import datetime, date


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with French formatting."""
    if value == int(value):
        return f"{int(value):,}".replace(",", " ")
    return f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",")


def calc_heures_supplementaires(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate overtime pay.
    
    Inputs:
        - salaire_mensuel: Monthly salary
        - heures_25: Hours at 25% rate (workdays 6h-21h)
        - heures_50_nuit: Hours at 50% rate (workdays 21h-6h)
        - heures_50_repos: Hours at 50% rate (rest days 6h-21h)
        - heures_100: Hours at 100% rate (rest days 21h-6h)
    
    Base: 191 hours/month
    """
    salaire = float(inputs.get('salaire_mensuel', 0))
    h25 = float(inputs.get('heures_25', 0))
    h50_nuit = float(inputs.get('heures_50_nuit', 0))
    h50_repos = float(inputs.get('heures_50_repos', 0))
    h100 = float(inputs.get('heures_100', 0))
    
    if salaire <= 0:
        return {
            'success': False,
            'error': 'Le salaire mensuel doit être supérieur à 0',
            'result': {},
            'table': [],
            'notes': []
        }
    
    # Hourly rate (191 hours/month base)
    taux_horaire = salaire / 191
    
    # Calculate overtime
    mont_25 = h25 * taux_horaire * 1.25
    mont_50_nuit = h50_nuit * taux_horaire * 1.50
    mont_50_repos = h50_repos * taux_horaire * 1.50
    mont_100 = h100 * taux_horaire * 2.00
    
    mont_50_total = mont_50_nuit + mont_50_repos
    total = mont_25 + mont_50_total + mont_100
    
    table = [
        {'label': 'Salaire mensuel', 'value': f"{format_number(salaire)} DH"},
        {'label': 'Taux horaire (base 191h)', 'value': f"{format_number(taux_horaire)} DH/h"},
    ]
    
    if h25 > 0:
        table.append({'label': f'Heures à 25% ({h25}h)', 'value': f"{format_number(mont_25)} DH"})
    if mont_50_total > 0:
        table.append({'label': f'Heures à 50% ({h50_nuit + h50_repos}h)', 'value': f"{format_number(mont_50_total)} DH"})
    if h100 > 0:
        table.append({'label': f'Heures à 100% ({h100}h)', 'value': f"{format_number(mont_100)} DH"})
    
    table.append({'label': 'TOTAL heures supplémentaires', 'value': f"{format_number(total)} DH"})
    
    return {
        'success': True,
        'result': {
            'salaire_mensuel': salaire,
            'taux_horaire': round(taux_horaire, 2),
            'heures_25': h25,
            'heures_50': h50_nuit + h50_repos,
            'heures_100': h100,
            'montant_25': round(mont_25, 2),
            'montant_50': round(mont_50_total, 2),
            'montant_100': round(mont_100, 2),
            'total': round(total, 2),
        },
        'table': table,
        'notes': [
            "Base de calcul: 191 heures par mois.",
            "25%: jours ouvrables entre 6h et 21h",
            "50%: jours ouvrables 21h-6h ou repos/fériés 6h-21h",
            "100%: jours de repos/fériés entre 21h et 6h"
        ]
    }


def calc_conges_payes(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate paid leave entitlement.
    
    Inputs:
        - date_naissance: Birth date (DD/MM/YYYY)
        - date_embauche: Hire date (DD/MM/YYYY)
        - fonction: 1=Employee, 2=Executive
    
    Rules:
        - 1.5 days/month for adults, 2 days/month for minors
        - +1.5 days per 5 years of seniority
        - Maximum 30 working days
        - Minimum 6 months service required
    """
    date_naissance_str = inputs.get('date_naissance', '')
    date_embauche_str = inputs.get('date_embauche', '')
    fonction = int(inputs.get('fonction', 1))
    
    try:
        date_naissance = datetime.strptime(date_naissance_str, "%d/%m/%Y")
        date_embauche = datetime.strptime(date_embauche_str, "%d/%m/%Y")
    except ValueError:
        return {
            'success': False,
            'error': 'Format de date invalide. Utilisez JJ/MM/AAAA',
            'result': {},
            'table': [],
            'notes': []
        }
    
    today = datetime.now()
    
    # Calculate age
    age = (today - date_naissance).days // 365
    
    # Calculate seniority
    anciennete_jours = (today - date_embauche).days
    anciennete_mois = anciennete_jours // 30
    anciennete_annees = anciennete_jours // 365
    
    # Check eligibility (6 months minimum)
    if anciennete_mois < 6:
        return {
            'success': True,
            'result': {'eligible': False, 'anciennete_mois': anciennete_mois},
            'table': [
                {'label': 'Ancienneté', 'value': f"{anciennete_mois} mois"},
                {'label': 'Éligibilité', 'value': 'Non éligible'},
            ],
            'notes': [
                "Le droit au congé s'acquiert après 6 mois de service continu.",
                f"Il vous reste {6 - anciennete_mois} mois avant d'être éligible."
            ]
        }
    
    # Base leave per month
    jours_par_mois = 2 if age < 18 else 1.5
    
    # Annual base leave
    conges_base = jours_par_mois * 12
    
    # Seniority bonus (1.5 days per 5 years)
    bonus_anciennete = (anciennete_annees // 5) * 1.5
    
    # Total (max 30 days)
    conges_total = min(conges_base + bonus_anciennete, 30)
    
    return {
        'success': True,
        'result': {
            'eligible': True,
            'age': age,
            'anciennete_annees': anciennete_annees,
            'anciennete_mois': anciennete_mois,
            'jours_par_mois': jours_par_mois,
            'conges_base': conges_base,
            'bonus_anciennete': bonus_anciennete,
            'conges_total': conges_total,
        },
        'table': [
            {'label': 'Âge', 'value': f"{age} ans"},
            {'label': 'Ancienneté', 'value': f"{anciennete_annees} ans et {anciennete_mois % 12} mois"},
            {'label': 'Fonction', 'value': 'Cadre' if fonction == 2 else 'Employé'},
            {'label': 'Congés de base', 'value': f"{conges_base} jours/an"},
            {'label': 'Bonus ancienneté', 'value': f"+{bonus_anciennete} jours"},
            {'label': 'TOTAL congés payés', 'value': f"{conges_total} jours ouvrables"},
        ],
        'notes': [
            f"Base: {jours_par_mois} jours par mois pour les {'moins' if age < 18 else 'plus'} de 18 ans.",
            "Majoration: 1,5 jours par période de 5 ans d'ancienneté.",
            "Maximum: 30 jours ouvrables."
        ]
    }


def calc_amortissement_lineaire(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate linear depreciation schedule.
    
    Inputs:
        - type_bien: Asset type (1-9)
        - prix_acquisition: Purchase price
        - mois_acquisition: Acquisition month (1-12)
        - annee_acquisition: Acquisition year
    
    Types:
        1: Buildings (4%, 25 years)
        2: Fixtures (5%, 20 years)
        3: Equipment (10%, 10 years)
        4: Office furniture (10%, 10 years)
        5: IT equipment (10%, 10 years)
        6: Transport equipment (20%, 5 years)
        7: Vehicles (20%, 5 years)
        8: Passenger cars (20%, 5 years, capped at 60000 DH/year)
        9: Software (30%, 3-4 years)
    """
    type_bien = int(inputs.get('type_bien', 3))
    prix = float(inputs.get('prix_acquisition', 0))
    mois = int(inputs.get('mois_acquisition', 1))
    annee = int(inputs.get('annee_acquisition', datetime.now().year))
    
    if prix <= 0:
        return {
            'success': False,
            'error': 'Le prix d\'acquisition doit être supérieur à 0',
            'result': {},
            'table': [],
            'notes': []
        }
    
    # Rates by type
    rates = {
        1: {'rate': 4, 'label': 'Constructions (25 ans)'},
        2: {'rate': 5, 'label': 'Agencements (20 ans)'},
        3: {'rate': 10, 'label': 'Matériel et outillage (10 ans)'},
        4: {'rate': 10, 'label': 'Mobilier de bureau (10 ans)'},
        5: {'rate': 10, 'label': 'Matériel informatique (10 ans)'},
        6: {'rate': 20, 'label': 'Matériel de transport (5 ans)'},
        7: {'rate': 20, 'label': 'Matériel roulant (5 ans)'},
        8: {'rate': 20, 'label': 'Voiture de tourisme (5 ans)'},
        9: {'rate': 30, 'label': 'Logiciels (3-4 ans)'},
    }
    
    if type_bien not in rates:
        type_bien = 3
    
    rate = rates[type_bien]['rate'] / 100
    label = rates[type_bien]['label']
    duree = int(1 / rate)
    
    # Calculate depreciation schedule
    schedule = []
    annee_courante = annee
    cumul = 0
    
    # First year (prorata)
    mois_restants = 12 - (mois - 1)
    amort_1 = prix * rate * mois_restants / 12
    cumul += amort_1
    schedule.append({
        'annee': annee_courante,
        'amortissement': round(amort_1, 2),
        'cumul': round(cumul, 2)
    })
    annee_courante += 1
    
    # Full years
    for _ in range(duree - 1):
        amort = prix * rate
        cumul += amort
        schedule.append({
            'annee': annee_courante,
            'amortissement': round(amort, 2),
            'cumul': round(cumul, 2)
        })
        annee_courante += 1
    
    # Last year (prorata)
    mois_derniere = mois - 1
    if mois_derniere > 0:
        amort_last = prix * rate * mois_derniere / 12
        cumul += amort_last
        schedule.append({
            'annee': annee_courante,
            'amortissement': round(amort_last, 2),
            'cumul': round(cumul, 2)
        })
    
    table = [
        {'label': 'Type d\'immobilisation', 'value': label},
        {'label': 'Prix d\'acquisition', 'value': f"{format_number(prix)} DH"},
        {'label': 'Date d\'acquisition', 'value': f"{mois:02d}/{annee}"},
        {'label': 'Taux d\'amortissement', 'value': f"{rates[type_bien]['rate']}%"},
        {'label': 'Durée', 'value': f"{duree} ans"},
    ]
    
    # Add first 3 years to table
    for i, row in enumerate(schedule[:3]):
        table.append({
            'label': f"Année {row['annee']}", 
            'value': f"{format_number(row['amortissement'])} DH"
        })
    
    if len(schedule) > 3:
        table.append({'label': '...', 'value': f"(voir tableau complet)"})
    
    notes = [
        "Amortissement calculé au prorata temporis pour l'année d'acquisition."
    ]
    if type_bien == 8:
        notes.append("Pour les voitures de tourisme, l'amortissement fiscal est plafonné à 60 000 DH/an.")
    
    return {
        'success': True,
        'result': {
            'type_bien': type_bien,
            'type_label': label,
            'prix_acquisition': prix,
            'taux': rates[type_bien]['rate'],
            'duree': duree,
            'schedule': schedule,
        },
        'table': table,
        'notes': notes
    }


def calc_auto_entrepreneur(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate auto-entrepreneur tax.
    
    Inputs:
        - chiffre_affaires: Annual revenue
        - type_activite: Activity type (1=commercial, 2=artisanal, 3=services)
    
    Rates:
        - Commercial/industrial: 0.5% (ceiling 500,000 DH)
        - Artisanal: 1% (ceiling 500,000 DH)
        - Services: 2% (ceiling 200,000 DH)
    """
    ca = float(inputs.get('chiffre_affaires', 0))
    activite = int(inputs.get('type_activite', 1))
    
    if ca <= 0:
        return {
            'success': False,
            'error': 'Le chiffre d\'affaires doit être supérieur à 0',
            'result': {},
            'table': [],
            'notes': []
        }
    
    rates = {
        1: {'taux': 0.5, 'label': 'Activités commerciales et industrielles', 'plafond': 500000},
        2: {'taux': 1.0, 'label': 'Activités artisanales', 'plafond': 500000},
        3: {'taux': 2.0, 'label': 'Prestations de services', 'plafond': 200000},
    }
    
    if activite not in rates:
        activite = 1
    
    taux = rates[activite]['taux']
    label = rates[activite]['label']
    plafond = rates[activite]['plafond']
    
    depasse_plafond = ca > plafond
    impot = ca * taux / 100
    impot_mensuel = impot / 12
    
    table = [
        {'label': 'Chiffre d\'affaires', 'value': f"{format_number(ca)} DH"},
        {'label': 'Type d\'activité', 'value': label},
        {'label': 'Plafond de CA', 'value': f"{format_number(plafond)} DH"},
        {'label': 'Taux d\'imposition', 'value': f"{taux}%"},
        {'label': 'Impôt annuel', 'value': f"{format_number(impot)} DH"},
        {'label': 'Impôt mensuel', 'value': f"{format_number(impot_mensuel)} DH"},
    ]
    
    notes = [
        "Le régime auto-entrepreneur est un régime simplifié.",
        "L'impôt est calculé sur le CA encaissé (et non le bénéfice)."
    ]
    
    if depasse_plafond:
        notes.insert(0, f"⚠️ ATTENTION: Le CA dépasse le plafond ({format_number(plafond)} DH). Vous devez opter pour un autre régime fiscal.")
    
    return {
        'success': True,
        'result': {
            'chiffre_affaires': ca,
            'type_activite': activite,
            'type_label': label,
            'taux': taux,
            'plafond': plafond,
            'depasse_plafond': depasse_plafond,
            'impot_annuel': round(impot, 2),
            'impot_mensuel': round(impot_mensuel, 2),
        },
        'table': table,
        'notes': notes
    }


def calc_sanctions_retard(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate late filing/payment penalties.
    
    Inputs:
        - montant: Tax amount
        - mois_retard: Months of delay
        - type_retard: 1=late filing, 2=late payment
    
    Rules:
        - First month: 5%
        - Beyond: +0.5% per additional month
        - Minimum: 500 DH (or 100 DH for some taxes)
    """
    montant = float(inputs.get('montant', 0))
    mois = int(inputs.get('mois_retard', 1))
    type_retard = int(inputs.get('type_retard', 2))
    
    if montant <= 0:
        return {
            'success': False,
            'error': 'Le montant doit être supérieur à 0',
            'result': {},
            'table': [],
            'notes': []
        }
    
    # Calculate penalties
    majoration = 0
    if mois >= 1:
        majoration = montant * 0.05  # 5% first month
        if mois > 1:
            majoration += montant * 0.005 * (mois - 1)  # 0.5% per additional month
    
    # Minimum penalty
    minimum = 500
    if majoration > 0 and majoration < minimum:
        majoration = minimum
    
    total = montant + majoration
    
    type_label = "dépôt tardif" if type_retard == 1 else "paiement tardif"
    
    return {
        'success': True,
        'result': {
            'montant_principal': montant,
            'mois_retard': mois,
            'type_retard': type_label,
            'majorations': round(majoration, 2),
            'total': round(total, 2),
        },
        'table': [
            {'label': 'Montant principal', 'value': f"{format_number(montant)} DH"},
            {'label': 'Mois de retard', 'value': f"{mois} mois"},
            {'label': 'Type', 'value': type_label.capitalize()},
            {'label': 'Majorations', 'value': f"{format_number(majoration)} DH"},
            {'label': 'Total à payer', 'value': f"{format_number(total)} DH"},
        ],
        'notes': [
            "Majoration: 5% pour le 1er mois + 0,5% par mois supplémentaire.",
            "Majoration minimale: 500 DH."
        ]
    }
