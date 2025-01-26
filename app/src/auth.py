import streamlit as st
from passlib.hash import pbkdf2_sha256
from datetime import datetime
from .utils import logger

def init_auth():
    """Initialize authentication state"""
    if 'user' not in st.session_state:
        st.session_state.user = None
    logger.info("Authentication state initialized")

def hash_password(password: str) -> str:
    """Hash a password using PBKDF2"""
    return pbkdf2_sha256.hash(password)

def verify_password(password: str, hash: str) -> bool:
    """Verify a password against its hash"""
    return pbkdf2_sha256.verify(password, hash)

def create_user(session, username: str, password: str, major: str) -> bool:
    """Create a new user"""
    from .models import User
    
    try:
        # Check if username already exists
        if session.query(User).filter_by(username=username).first():
            logger.warning("Attempted to create duplicate user: %s", username)
            return False
        
        # Create new user
        user = User(
            username=username,
            password_hash=hash_password(password),
            major=major,
            created_at=datetime.utcnow()
        )
        session.add(user)
        session.commit()
        
        logger.info("New user created successfully: %s", username)
        return True
        
    except Exception as e:
        session.rollback()
        logger.error("Failed to create user: %s - Error: %s", username, str(e))
        return False

def login_user(session, username: str, password: str) -> bool:
    """Log in a user"""
    from .models import User
    
    try:
        # Log the login attempt
        logger.info("Login attempt for username: %s", username)
        
        # Query the user
        user = session.query(User).filter_by(username=username).first()
        
        if not user:
            logger.warning("User not found: %s", username)
            return False
            
        logger.debug("Found user: %s, verifying password", username)
        
        try:
            is_valid = verify_password(password, user.password_hash)
            logger.debug("Password verification result: %s", is_valid)
        except Exception as e:
            logger.error("Password verification failed: %s", str(e))
            return False
            
        if not is_valid:
            logger.warning("Invalid password for user: %s", username)
            return False
        
        # Store user info in session state
        st.session_state.user = {
            'id': user.id,
            'username': user.username,
            'major': user.major
        }
        
        logger.info("User logged in successfully: %s", username)
        return True
        
    except Exception as e:
        session.rollback()
        logger.error("Login error for user: %s - Error: %s", username, str(e))
        return False

def logout_user():
    """Log out the current user"""
    if st.session_state.user:
        username = st.session_state.user['username']
        st.session_state.user = None
        logger.info("User logged out: %s", username)

def is_logged_in() -> bool:
    """Check if a user is currently logged in"""
    return st.session_state.user is not None

def get_current_user() -> dict:
    """Get the current user's information"""
    return st.session_state.user 