from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session

# --- SQLAlchemy Setup ---
Base = declarative_base()

# --- SQLAlchemy ORM Models (Schema Definition) ---
class User(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    user_email = Column(String, unique=True, nullable=False)
    user_name = Column(String, nullable=False)
    timestamp = Column(Float, nullable=False)

class File(Base):
    __tablename__ = "files"
    id = Column(Integer, primary_key=True, autoincrement=True)
    file_hash = Column(String, unique=True, nullable=False)
    storage_path = Column(String, nullable=False)
    creator_user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    timestamp = Column(Float, nullable=False)
    creator = relationship("User")

class Commit(Base):
    __tablename__ = "commits"
    id = Column(Integer, primary_key=True, autoincrement=True)
    file_hash = Column(String, ForeignKey("files.file_hash"), nullable=False)
    actual_filename = Column(String, nullable=False)
    committer_user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    timestamp = Column(Float, nullable=False)
    committer = relationship("User")
    file = relationship("File")

class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    id = Column(Integer, primary_key=True, autoincrement=True)
    commit_id = Column(Integer, ForeignKey("commits.id"), unique=True, nullable=False)
    food_item = Column(String)
    calories = Column(Integer)
    ingredients_json = Column(String)
    timestamp = Column(Float, nullable=False)
    commit = relationship("Commit")

class NutritionFact(Base):
    __tablename__ = "nutrition_facts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    # The 'name' is the key we will search against.
    name = Column(String, unique=True, nullable=False, index=True)
    # All values are per 100g
    calories = Column(Float, nullable=False)
    fat = Column(Float, nullable=False)
    saturated_fat = Column(Float)
    carbohydrates = Column(Float, nullable=False)
    sugar = Column(Float)
    protein = Column(Float, nullable=False)
    salt = Column(Float) # Salt in mg per 100g