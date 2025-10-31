import os
import uuid
import logging
import hashlib
import time
import json
from bloom_filter import BloomFilter

from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session

from config import settings
from db_classes import Base, User, File, Commit, AnalysisResult, NutritionFact

# --- Configuration ---
IMAGE_DIR = "images"
DB_PATH = "files.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# System user details
SYSTEM_USER_EMAIL = "system@innox.app"
SYSTEM_USER_NAME = "system"

# --- Bloom Filter Configuration ---
BLOOM_CAPACITY = settings.BLOOMFILTER_SIZE
BLOOM_FPR = settings.BLOOMFILTER_FPR
FILE_HASH_CACHE = BloomFilter(max_elements=BLOOM_CAPACITY, error_rate=BLOOM_FPR)

logger = logging.getLogger("app.storage")
logger.setLevel(logging.INFO)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class FileHandler:
    """Handles all direct interactions with the physical file system."""
    def __init__(self, image_dir: str):
        self.image_dir = image_dir
        os.makedirs(self.image_dir, exist_ok=True)

    @staticmethod
    def calculate_hash(image_bytes: bytes) -> str:
        """Calculates the SHA-256 hash of the image bytes."""
        return hashlib.sha256(image_bytes).hexdigest()

    def save_to_disk(self, image_bytes: bytes, original_filename: str) -> str:
        """Saves image bytes to a unique path on disk and returns the path."""
        file_extension = os.path.splitext(original_filename)[1] or ".jpg"
        unique_name = f"{uuid.uuid4()}{file_extension}"
        storage_path = os.path.join(self.image_dir, unique_name)
        with open(storage_path, "wb") as buffer:
            buffer.write(image_bytes)
        logger.info(f"New file saved to disk: {storage_path}")
        return storage_path

class DatabaseHandler:
    """Handles all interactions with the SQLAlchemy database session."""
    def __init__(self, db_session: Session):
        self.db = db_session

    def get_system_user(self) -> User:
        """Retrieves the system user object from the database."""
        user = self.db.query(User).filter(User.user_email == SYSTEM_USER_EMAIL).first()
        if not user:
            raise RuntimeError("System user not found. DB not properly initialized.")
        return user

    def find_file_by_hash(self, file_hash: str) -> File | None:
        """Finds a file record by its hash, checking the cache first."""
        if file_hash not in FILE_HASH_CACHE:
            return None
        return self.db.query(File).filter(File.file_hash == file_hash).first()

    def find_nutrition_fact(self, food_name: str) -> NutritionFact | None:
        """
        Performs a fuzzy search to find a nutrition record.
        This is a simple implementation; more complex fuzzy matching could be used.
        """
        # Search for a record where the name is contained within the LLM's identified item.
        # e.g., DB 'chicken breast' will match LLM 'grilled chicken breast'
        return self.db.query(NutritionFact).filter(NutritionFact.name.like(f"%{food_name.lower()}%")).first()

    def insert_commit_and_analysis(self, file_hash: str, actual_filename: str, analysis_data: dict):
        """Creates and adds Commit and AnalysisResult records to the session."""
        system_user = self.get_system_user()
        current_time = time.time()
        new_commit = Commit(
            file_hash=file_hash,
            actual_filename=actual_filename,
            committer_user_id=system_user.user_id,
            timestamp=current_time,
        )
        self.db.add(new_commit)
        self.db.flush()  # Ensures new_commit.id is available for the foreign key

        new_analysis = AnalysisResult(
            commit_id=new_commit.id,
            food_item=analysis_data.get("food_item"),
            calories=analysis_data.get("calories"),
            ingredients_json=json.dumps(analysis_data.get("ingredients", {})),
            timestamp=current_time,
        )
        self.db.add(new_analysis)
        logger.info(f"Commit ({new_commit.id}) and analysis records added to session.")

    def insert_file(self, file_hash: str, storage_path: str):
        """Creates and adds a new File record to the session."""
        system_user = self.get_system_user()
        new_file = File(
            file_hash=file_hash,
            storage_path=storage_path,
            creator_user_id=system_user.user_id,
            timestamp=time.time(),
        )
        self.db.add(new_file)
        FILE_HASH_CACHE.add(file_hash) # Update cache
        logger.info(f"New file record for {storage_path} added to session.")

# --- Transaction Orchestrator ---

class StorageTransactionManager:
    """Orchestrates the file and database operations for a single transaction."""
    def __init__(self, db_session: Session):
        self.db_handler = DatabaseHandler(db_session)
        self.file_handler = FileHandler(IMAGE_DIR)

    def execute(self, image_bytes: bytes, original_filename: str, analysis_data: dict):
        """Executes the full save-and-analyze process."""
        file_hash = self.file_handler.calculate_hash(image_bytes)
        existing_file = self.db_handler.find_file_by_hash(file_hash)

        if existing_file:
            logger.info(f"Storage: Cache hit for existing file: {existing_file.storage_path}")
            self.db_handler.insert_commit_and_analysis(file_hash, original_filename, analysis_data)
        else:
            storage_path = self.file_handler.save_to_disk(image_bytes, original_filename)
            self.db_handler.insert_file(file_hash, storage_path)
            self.db_handler.insert_commit_and_analysis(file_hash, original_filename, analysis_data)

# --- Public-Facing Functions ---

def initialize_database():
    """Initializes the database, tables, and system user."""
    os.makedirs(IMAGE_DIR, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.user_email == SYSTEM_USER_EMAIL).first()
        if not user:
            db.add(User(email=SYSTEM_USER_EMAIL, name=SYSTEM_USER_NAME, timestamp=time.time()))
            db.commit()
            logger.info("Database initialized. System user created.")
        else:
            logger.info("Database initialized. System user already exists.")

        files = db.query(File.file_hash).all()
        for file in files:
            FILE_HASH_CACHE.add(file.file_hash)
        logger.info(f"Bloom Filter primed with {len(files)} existing file hashes.")
    except Exception as e:
        logger.error(f"FATAL: Database initialization failed: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def save_image_and_analysis(image_bytes: bytes, original_filename: str, analysis_data: dict):
    """Public function to save image and analysis, managing the session and transaction."""
    db = SessionLocal()
    try:
        manager = StorageTransactionManager(db)
        manager.execute(image_bytes, original_filename, analysis_data)
        db.commit()
        logger.info("Database transaction committed successfully.")
    except Exception as e:
        logger.error(f"Database transaction failed: {e}")
        db.rollback()
    finally:
        db.close()
