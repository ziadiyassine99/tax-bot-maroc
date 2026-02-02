"""
Calculator UI components for Streamlit.
"""

import streamlit as st
from typing import Dict, Any, List, Optional
from .registry import get_calculator, execute_calculator


def render_calculator_card(calculator_id: str, expanded: bool = True) -> Optional[Dict[str, Any]]:
    """
    Render an interactive calculator card in the chat.
    
    Args:
        calculator_id: ID of the calculator to render
        expanded: Whether to expand the calculator by default
    
    Returns:
        Calculation result if submitted, None otherwise
    """
    calculator = get_calculator(calculator_id)
    if not calculator:
        st.error(f"Calculateur non trouvé: {calculator_id}")
        return None
    
    # Custom styling for calculator cards
    st.markdown("""
        <style>
        .calculator-card {
            background: linear-gradient(135deg, #F0F7FF 0%, #E8F4FD 100%);
            border: 2px solid #4A90D9;
            border-radius: 16px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        .calc-title {
            color: #2C5282;
            font-weight: 600;
            font-size: 1.1rem;
        }
        .calc-result {
            background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
            border: 2px solid #4CAF50;
            border-radius: 12px;
            padding: 1rem;
            margin-top: 0.5rem;
        }
        .result-row {
            display: flex;
            justify-content: space-between;
            padding: 0.25rem 0;
            border-bottom: 1px dashed #A5D6A7;
        }
        .result-label {
            color: #2E7D32;
        }
        .result-value {
            font-weight: 600;
            color: #1B5E20;
        }
        </style>
    """, unsafe_allow_html=True)
    
    with st.expander(f"{calculator['icon']} {calculator['title']}", expanded=expanded):
        st.markdown(f"*{calculator['description']}*")
        
        # Create form
        with st.form(key=f"calc_form_{calculator_id}"):
            inputs = {}
            
            # Render input fields based on field definitions
            for field in calculator.get('fields', []):
                field_id = field['id']
                field_label = field['label']
                field_type = field.get('type', 'number')
                default_value = field.get('default', '')
                required = field.get('required', False)
                
                if field_type == 'number':
                    inputs[field_id] = st.number_input(
                        field_label,
                        value=float(default_value) if default_value else 0.0,
                        key=f"{calculator_id}_{field_id}"
                    )
                elif field_type == 'text':
                    inputs[field_id] = st.text_input(
                        field_label,
                        value=str(default_value) if default_value else '',
                        key=f"{calculator_id}_{field_id}"
                    )
                elif field_type == 'select':
                    options = field.get('options', [])
                    option_labels = [opt['label'] for opt in options]
                    option_values = [opt['value'] for opt in options]
                    
                    # Find default index
                    default_idx = 0
                    if default_value in option_values:
                        default_idx = option_values.index(default_value)
                    
                    selected_label = st.selectbox(
                        field_label,
                        options=option_labels,
                        index=default_idx,
                        key=f"{calculator_id}_{field_id}"
                    )
                    # Map back to value
                    selected_idx = option_labels.index(selected_label)
                    inputs[field_id] = option_values[selected_idx]
            
            # Submit button
            submitted = st.form_submit_button("🧮 Calculer", use_container_width=True)
            
            if submitted:
                # Execute calculation
                result = execute_calculator(calculator_id, inputs)
                
                if result.get('success', False):
                    # Display results
                    render_calculator_results(result)
                    return result
                else:
                    st.error(result.get('error', 'Erreur de calcul'))
                    return None
    
    return None


def render_calculator_results(result: Dict[str, Any]):
    """
    Render calculation results in a nice format.
    """
    st.markdown("---")
    st.markdown("### 📊 Résultats")
    
    # Render result table
    table = result.get('table', [])
    if table:
        for row in table:
            label = row.get('label', '')
            value = row.get('value', '')
            
            # Check if this is a "total" or important row
            is_important = any(keyword in label.lower() for keyword in ['total', 'final', 'net', 'résultat'])
            
            if is_important:
                st.markdown(f"**{label}**: **{value}**")
            else:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(label)
                with col2:
                    st.write(value)
    
    # Render notes
    notes = result.get('notes', [])
    if notes:
        st.markdown("---")
        st.markdown("**💡 Notes:**")
        for note in notes:
            st.markdown(f"- {note}")


def render_calculator_suggestion(calculator_info: Dict[str, Any]) -> bool:
    """
    Render a small suggestion card for a calculator.
    Returns True if user clicks to expand.
    """
    icon = calculator_info.get('icon', '🧮')
    title = calculator_info.get('title', 'Calculateur')
    description = calculator_info.get('description', '')
    calc_id = calculator_info.get('id', '')
    
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #FFF8E1 0%, #FFECB3 100%);
            border: 1px solid #FFB300;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            margin: 0.5rem 0;
        ">
            <span style="font-size: 1.2rem;">{icon}</span>
            <strong>{title}</strong>
            <br>
            <small style="color: #666;">{description}</small>
        </div>
    """, unsafe_allow_html=True)
    
    return st.button(f"Ouvrir {title}", key=f"suggest_{calc_id}")


def render_inline_calculator(calculator_id: str, context: str = "") -> Optional[Dict[str, Any]]:
    """
    Render a calculator inline in the chat flow with context.
    
    Args:
        calculator_id: Calculator to render
        context: Optional context message to show above
    
    Returns:
        Result dict if calculation performed, None otherwise
    """
    if context:
        st.info(f"💡 {context}")
    
    return render_calculator_card(calculator_id, expanded=True)


def render_calculator_list_by_category(category: str):
    """
    Render a list of calculators for a given category.
    Useful for a dedicated calculators page.
    """
    from .registry import get_calculators_by_category
    
    calculators = get_calculators_by_category(category)
    
    category_titles = {
        'social': '🏦 Calculateurs Social / CNSS',
        'fiscal': '📋 Calculateurs Fiscal',
        'ratios': '📊 Ratios Financiers'
    }
    
    st.markdown(f"## {category_titles.get(category, category.title())}")
    
    cols = st.columns(2)
    for idx, calc in enumerate(calculators):
        with cols[idx % 2]:
            if st.button(
                f"{calc['icon']} {calc['title']}",
                key=f"list_{calc['id']}",
                use_container_width=True
            ):
                st.session_state[f"open_calc_{calc['id']}"] = True
                st.rerun()
            st.caption(calc['description'])
