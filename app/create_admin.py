from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.models import User
from src.auth import hash_password
from src.utils import logger

def create_admin_user(username: str, password: str):
    """Create an admin user in the database"""
    try:
        # Initialize database connection
        engine = create_engine('sqlite:///annotation.db')
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Check if admin user already exists
        existing_user = session.query(User).filter_by(username=username).first()
        if existing_user:
            logger.warning(f"User {username} already exists")
            return False
        
        # Create new admin user
        password_hash = hash_password(password)
        admin_user = User(
            username=username,
            password_hash=password_hash,
            major="admin",
            is_active=True
        )
        
        # Add to database
        session.add(admin_user)
        session.commit()
        logger.info(f"Created admin user: {username}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create admin user: {str(e)}")
        return False

if __name__ == "__main__":
    # Create admin user with username "admin" and password "admin"
    success = create_admin_user("admin", "admin")
    if success:
        print("Admin user created successfully!")
    else:
        print("Failed to create admin user. Check logs for details.") 