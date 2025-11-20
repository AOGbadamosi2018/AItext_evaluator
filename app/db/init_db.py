"""Initialize the database by creating all tables."""
import asyncio
from .session import init_db
from app.services.evaluation_service import evaluation_service


async def init():
    """Initialize the database and services."""
    print("Initializing database...")
    init_db()
    print("Database initialized successfully!")
    
    # Initialize evaluation service
    print("Initializing evaluation services...")
    await evaluation_service.initialize()
    print("Evaluation services initialized successfully!")


def main():
    """Run the initialization."""
    asyncio.run(init())


if __name__ == "__main__":
    main()
