"""
Authentication module for IYYA Legal Assistant.
Handles user authentication with hashed passwords.
"""

import hashlib
import json
import os
from typing import Optional, Dict
from datetime import datetime

import streamlit as st


# =============================================================================
# Password Hashing
# =============================================================================

# Fixed salt for password hashing (in production, use unique salt per user)
SALT = "iyya_legal_assistant_2026"


def hash_password(password: str) -> str:
    """
    Hash a password using SHA-256 with salt.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password string
    """
    salted = f"{SALT}{password}{SALT}"
    return hashlib.sha256(salted.encode()).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        password: Plain text password to verify
        password_hash: Stored hash to compare against
        
    Returns:
        True if password matches, False otherwise
    """
    return hash_password(password) == password_hash


# =============================================================================
# User Management
# =============================================================================

USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")


def load_users() -> Dict:
    """
    Load users from the JSON file.
    
    Returns:
        Dictionary containing user data
    """
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"users": {}}
    except json.JSONDecodeError:
        return {"users": {}}


def authenticate(username: str, password: str) -> bool:
    """
    Authenticate a user with username and password.
    
    Args:
        username: The username
        password: The plain text password
        
    Returns:
        True if authentication successful, False otherwise
    """
    users_data = load_users()
    users = users_data.get("users", {})
    
    if username not in users:
        return False
    
    stored_hash = users[username].get("password_hash", "")
    return verify_password(password, stored_hash)


# =============================================================================
# Login Page UI
# =============================================================================

def render_login_page():
    """Render the login page with golden theme styling."""
    
    # Custom CSS for login page
    st.markdown("""
        <style>
        .login-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 2rem;
            background: linear-gradient(135deg, #FFF8EC 0%, #F5EBD7 100%);
            border: 2px solid #D4A574;
            border-radius: 20px;
            box-shadow: 0 8px 30px rgba(139, 105, 20, 0.2);
        }
        
        .login-title {
            font-family: 'Playfair Display', serif;
            font-size: 2.5rem;
            font-weight: 700;
            color: #8B6914;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        
        .login-subtitle {
            font-family: 'Inter', sans-serif;
            color: #6B5A3E;
            text-align: center;
            font-size: 1rem;
            margin-bottom: 2rem;
        }
        
        .stTextInput > div > div > input {
            background-color: #FFF8EC !important;
            border: 2px solid #D4A574 !important;
            border-radius: 10px !important;
            color: #2D2A26 !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #8B6914 !important;
            box-shadow: 0 0 0 2px rgba(139, 105, 20, 0.2) !important;
        }
        
        .login-button button {
            background: linear-gradient(135deg, #8B6914 0%, #B8860B 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.75rem 2rem !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            width: 100% !important;
            transition: all 0.3s ease !important;
        }
        
        .login-button button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 15px rgba(139, 105, 20, 0.3) !important;
        }
        
        .error-message {
            background-color: #FFE4E4;
            border: 1px solid #FF6B6B;
            border-radius: 10px;
            padding: 1rem;
            color: #D63031;
            text-align: center;
            margin-top: 1rem;
            font-family: 'Inter', sans-serif;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("")
        st.markdown("")
        st.markdown("")
        
        # Title
        st.markdown('<h1 class="login-title">IYYA</h1>', unsafe_allow_html=True)
        st.markdown('<p class="login-subtitle">Assistant Juridique Marocain</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Login form
        with st.form("login_form"):
            username = st.text_input("Nom d'utilisateur", placeholder="Entrez votre identifiant")
            password = st.text_input("Mot de passe", type="password", placeholder="Entrez votre mot de passe")
            
            st.markdown("")
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                submitted = st.form_submit_button("Se connecter", use_container_width=True)
            
            if submitted:
                if username and password:
                    if authenticate(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.markdown('<div class="error-message">Identifiant ou mot de passe incorrect</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="error-message">Veuillez remplir tous les champs</div>', unsafe_allow_html=True)


def logout():
    """Log out the current user."""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.rerun()


def init_auth_state():
    """Initialize authentication session state variables."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None


def is_authenticated() -> bool:
    """Check if the user is authenticated."""
    return st.session_state.get("authenticated", False)
