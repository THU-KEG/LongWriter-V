from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from .utils import logger
from sqlalchemy import inspect

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    major = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    annotations = relationship("Annotation", back_populates="annotator")
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', major='{self.major}')>"

class Annotation(Base):
    __tablename__ = 'annotations'
    
    id = Column(Integer, primary_key=True)
    slide_id = Column(Integer, nullable=False)
    major = Column(String, nullable=False)
    course_name = Column(String, nullable=False)
    original_script = Column(String)
    modified_script = Column(String)
    is_completed = Column(Boolean, default=False)
    annotator_id = Column(Integer, ForeignKey('users.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    annotator = relationship("User", back_populates="annotations")

def init_db(db_url: str):
    """Initialize the database"""
    try:
        logger.info("Initializing database at: %s", db_url)
        engine = create_engine(db_url)
        
        # Only create tables that don't exist
        Base.metadata.create_all(engine, checkfirst=True)
        
        # Log existing tables
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        logger.info("Existing tables: %s", tables)
        
        return engine
    except Exception as e:
        logger.error("Failed to initialize database: %s", str(e))
        raise 