"""
Calculator module for IYYA Legal Assistant.
Provides interactive calculation tools for tax, social, and financial computations.
"""

from .registry import CALCULATORS, get_calculator, get_calculators_by_category
from .matcher import find_matching_calculators, should_show_calculator
from .ui import render_calculator_card, render_calculator_results

__all__ = [
    'CALCULATORS',
    'get_calculator',
    'get_calculators_by_category',
    'find_matching_calculators',
    'should_show_calculator',
    'render_calculator_card',
    'render_calculator_results',
]
