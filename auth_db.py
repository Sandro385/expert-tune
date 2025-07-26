"""
auth_db.py
-------------

This module encapsulates all database interactions for the Expert‑Tune
application.  It uses SQLite as a lightweight, file‑based database to
store registered users and their chat histories.  Two tables are
created on first run: one for user credentials and another for the
chat transcripts.  Helper functions abstract away the SQL so that
`app.py` can simply call `add_user`, `save_msg` and `load_history`.

The database filename can be customised by setting the `DB` constant
below.  When run on Render with a persistent disk, this file will
reside on that volume, ensuring chat history and user accounts are
preserved across deployments.
"""

import sqlite3
import os
import streamlit as st

# Name of the SQLite database file.  If you wish to store the
# database in a different location, adjust this constant.  When
# deploying on Render, the database file should live on a mounted
# persistent disk.
DB = os.getenv("DATABASE_FILE", "users.db")


def init_db() -> None:
    """Initialise the SQLite database.

    This function creates the `users` and `chat_history` tables if
    they do not already exist.  It should be invoked exactly once at
    application startup.
    """
    conn = sqlite3.connect(DB)
    # Create a table for storing user credentials.  The password is
    # stored as a SHA256 hash; see `app.py` for details on how it is
    # generated.  The username is the primary key to prevent
    # duplicates.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users(
            username TEXT PRIMARY KEY,
            password_hash TEXT
        );
        """
    )
    # Create a table for storing chat messages.  Each row records the
    # message author (role), the domain (context), the content of the
    # message and a timestamp.  The auto‑incrementing `id` column
    # ensures messages are returned in order.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            domain TEXT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()
    conn.close()


def add_user(username: str, pwd_hash: str) -> None:
    """Insert a new user into the database.

    If the username already exists, this function does nothing.

    Args:
        username: The desired username.
        pwd_hash: A SHA256 hash of the user's password.
    """
    conn = sqlite3.connect(DB)
    conn.execute(
        "INSERT OR IGNORE INTO users (username, password_hash) VALUES (?, ?)",
        (username, pwd_hash),
    )
    conn.commit()
    conn.close()


def save_msg(username: str, domain: str, role: str, content: str) -> None:
    """Persist a single chat message to the database.

    Args:
        username: The user who owns this conversation.
        domain: The selected domain (e.g. "იურისტი").
        role: Either "user" or "assistant".
        content: The message content.
    """
    conn = sqlite3.connect(DB)
    conn.execute(
        "INSERT INTO chat_history(username, domain, role, content) VALUES (?, ?, ?, ?)",
        (username, domain, role, content),
    )
    conn.commit()
    conn.close()


def load_history(username: str, domain: str):
    """Retrieve the chat history for a given user and domain.

    Args:
        username: The user whose history to retrieve.
        domain: The domain context to filter on.

    Returns:
        A list of dictionaries with keys "role" and "content", ordered
        by message ID ascending.
    """
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT role, content FROM chat_history WHERE username = ? AND domain = ? ORDER BY id",
        (username, domain),
    ).fetchall()
    conn.close()
    return [
        {"role": role, "content": content} for role, content in rows
    ]


def get_users():
    """Return all registered users and their password hashes.

    This helper queries the ``users`` table and returns a list of
    tuples containing the username and the corresponding password
    hash.  It is used by the application to build the
    ``credentials`` dictionary required by ``streamlit_authenticator``.

    Returns:
        List[Tuple[str, str]]: A list of ``(username, password_hash)``
        pairs.  If there are no users in the database, the list will
        be empty.
    """
    conn = sqlite3.connect(DB)
    rows = conn.execute("SELECT username, password_hash FROM users").fetchall()
    conn.close()
    return rows