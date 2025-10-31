# app/db_schema.py

DB_PATH = "files.db"

# --- Schema Definitions ---

# Users Table Schema
USER_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT UNIQUE NOT NULL,
        user_name TEXT NOT NULL,
        timestamp REAL NOT NULL
    )
"""

# Files Table Schema (The unique inventory of files)
FILES_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_hash TEXT UNIQUE NOT NULL,
        storage_path TEXT NOT NULL,
        creator_user_id INTEGER NOT NULL,
        timestamp REAL NOT NULL,
        FOREIGN KEY (creator_user_id) REFERENCES users(user_id)
    )
"""

# Commits Table Schema (The usage log for when a file was uploaded)
COMMITS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS commits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_hash TEXT NOT NULL,
        actual_filename TEXT NOT NULL,
        committer_user_id INTEGER NOT NULL,
        timestamp REAL NOT NULL,
        FOREIGN KEY (file_hash) REFERENCES files(file_hash),
        FOREIGN KEY (committer_user_id) REFERENCES users(user_id)
    )
"""