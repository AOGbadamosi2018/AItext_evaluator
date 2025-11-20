from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from app.core.config import settings

# Create database engine
engine = create_engine(
    settings.DATABASE_URL, connect_args={"check_same_thread": False}
)

# Create a scoped session factory
SessionLocal = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=engine)
)

# Dependency to get DB session
def get_db():
    """Get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create all tables
def init_db():
    """Initialize the database by creating all tables.
    
    This should be called during application startup.
    """
    from app.models.base import Base
    Base.metadata.create_all(bind=engine)
