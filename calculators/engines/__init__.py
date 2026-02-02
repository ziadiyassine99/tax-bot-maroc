"""
Calculation engines - Pure Python implementations of business logic.
"""

from .social import (
    calc_cotisations_cnss,
    calc_indemnite_maladie,
    calc_indemnite_maternite,
    calc_conge_naissance,
    calc_ipd_perte_emploi,
    calc_pension_invalidite,
)

from .fiscal import (
    calc_heures_supplementaires,
    calc_conges_payes,
    calc_amortissement_lineaire,
    calc_auto_entrepreneur,
    calc_sanctions_retard,
)

from .ratios import (
    calc_liquidite_generale,
    calc_liquidite_reduite,
    calc_marge_nette,
    calc_roe,
    calc_endettement_global,
    calc_autonomie_financiere,
)

__all__ = [
    # Social
    'calc_cotisations_cnss',
    'calc_indemnite_maladie',
    'calc_indemnite_maternite',
    'calc_conge_naissance',
    'calc_ipd_perte_emploi',
    'calc_pension_invalidite',
    # Fiscal
    'calc_heures_supplementaires',
    'calc_conges_payes',
    'calc_amortissement_lineaire',
    'calc_auto_entrepreneur',
    'calc_sanctions_retard',
    # Ratios
    'calc_liquidite_generale',
    'calc_liquidite_reduite',
    'calc_marge_nette',
    'calc_roe',
    'calc_endettement_global',
    'calc_autonomie_financiere',
]
