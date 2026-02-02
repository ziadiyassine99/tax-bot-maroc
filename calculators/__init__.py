"""
Calculator module for IYYA Legal Assistant.
Provides interactive calculation tools for tax, social, and financial computations.
"""

from .registry import CALCULATORS, get_calculator, get_calculators_by_category
from .matcher import find_matching_calculators, should_show_calculator, get_suggested_calculators
from .ui import render_calculator_card, render_calculator_results, render_calculator_suggestion

__all__ = [
    'CALCULATORS',
    'get_calculator',
    'get_calculators_by_category',
    'find_matching_calculators',
    'should_show_calculator',
    'get_suggested_calculators',
    'render_calculator_card',
    'render_calculator_results',
    'render_calculator_suggestion',
]
