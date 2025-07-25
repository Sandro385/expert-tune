"""
app.py
------

This Streamlit application wraps a lightweight question‑and‑answer
assistant that collects domain‑specific information from users and
stores the conversation context in a SQLite database.  It adds
registration and login functionality via `streamlit‑authenticator` and
persists chat history so that each user can resume conversations in
different domains.  When the user is ready, they can trigger a LoRA
fine‑tuning job that uses their stored chat history to build a
dataset for training.
"""

import os
import hashlib
import subprocess

import streamlit as st
import streamlit_authenticator as stauth
from openai import OpenAI

from auth_db import init_db, add_user, save_msg, load_history, get_users

# Initialise the SQLite database.  This creates the necessary tables
# on the first run.  If the database file resides on a persistent
# volume (e.g. Render's Disk), chat history and user accounts will
# survive across deployments.
init_db()

# Build the credentials dictionary from the SQLite users table.  The
# streamlit-authenticator package expects a nested dictionary of the form
# {"usernames": {username: {"name": <display_name>, "email": <email>, "password": <password_hash>}}}
def build_credentials():
    creds = {"usernames": {}}
    for uname, pwd_hash in get_users():
        # For this simple application we use the username for both the
        # display name and the email field.  If you wish to store
        # additional metadata (e.g. proper name or email) you can do
        # that in the database and populate it here.
        creds["usernames"][uname] = {
            "name": uname,
            "email": f"{uname}@example.com",
            "password": pwd_hash,
        }
    return creds


# -----------------------------------------------------------------------------
# 1. Authentication
# -----------------------------------------------------------------------------

# Streamlit‑authenticator maintains its own internal state in
# `st.session_state.auth`.  The first time the app loads, we create
# an authenticator instance with an empty credentials dictionary.
# Later, when users register, their credentials will be stored in
# SQLite, but for demonstration purposes we do not pre‑populate the
# authenticator.
if "auth" not in st.session_state:
    # Build a fresh credentials dictionary from the database.  This ensures
    # that any previously registered users are recognised by
    # streamlit‑authenticator.  The credentials dictionary must be
    # reconstructed at runtime because the authenticator stores it only in
    # memory and does not persist it across reruns.
    creds = build_credentials()
    st.session_state.auth = stauth.Authenticate(
        credentials=creds,
        cookie_name="expert_tune",
        cookie_key="abc123",
        cookie_expiry_days=30.0,
    )

# Render the login form in the sidebar.  The login call returns
# three values: name (unused), auth_status (True/False/None) and
# username.  We capture the username so that we can associate chat
# history with the current user.
name, auth_status, username = st.session_state.auth.login("sidebar")


# -----------------------------------------------------------------------------
# 2. Registration form
# -----------------------------------------------------------------------------

# Provide a simple registration form in the sidebar.  Clicking the
# "რეგისტრაცია" button reveals two input fields and a second button
# to submit the new account.  The user's password is hashed before
# being stored.
# Registration form (persist state across reruns).
if "register" not in st.session_state:
    st.session_state.register = False

if st.sidebar.button("რეგისტრაცია"):
    # Toggle the registration form flag on click
    st.session_state.register = not st.session_state.register

if st.session_state.register:
    # Persistent text inputs with keys so values survive reruns
    new_user = st.sidebar.text_input("მომხმარებელი", key="reg_user")
    new_pwd = st.sidebar.text_input("პაროლი", type="password", key="reg_pwd")
    if st.sidebar.button("დარეგისტრირდი", key="register_submit"):
        if new_user and new_pwd:
            pwd_hash = hashlib.sha256(new_pwd.encode()).hexdigest()
            # Persist the new user to the database
            add_user(new_user, pwd_hash)
            # Rebuild the authenticator with updated credentials.  The
            # Authenticate object does not expose a public ``credentials``
            # attribute in newer versions, so we cannot mutate it
            # directly.  Instead, recreate the authenticator using the
            # fresh credentials from the database.  This will also
            # reset any login state, requiring the user to log in again.
            new_creds = build_credentials()
            st.session_state.auth = stauth.Authenticate(
                credentials=new_creds,
                cookie_name="expert_tune",
                cookie_key="abc123",
                cookie_expiry_days=30.0,
            )
            st.sidebar.success("დარეგისტრირდით! გაიარეთ ლოგინი.")
            # Reset registration state after success
            st.session_state.register = False
            st.session_state.reg_user = ""
            st.session_state.reg_pwd = ""
        else:
            st.sidebar.warning("გთხოვთ შეავსოთ ორივე ველი.")


# -----------------------------------------------------------------------------
# 3. Main application logic
# -----------------------------------------------------------------------------

if auth_status:
    # If authenticated, provide a logout button in the sidebar
    st.session_state.auth.logout("sidebar")
    # Set a title for the page
    st.title("🎓 Expert‑Tune – რეგისტრაცია & ჩეთი")

    # Domain selection drop‑down.  The selected domain determines
    # which chat history to load and is stored in session_state for
    # later retrieval during fine‑tuning.
    domain = st.selectbox("სფერო", ["იურისტი", "ფსიქოლოგი", "რესტორატორი", "სხვა"])
    st.session_state.domain = domain

    # Construct a unique key for the current user's conversation in
    # this domain.  This allows multiple users and multiple domains
    # to have separate chat histories stored in session_state.
    key = f"msgs_{username}_{domain}"
    if key not in st.session_state:
        # Load any existing history from the database into session
        st.session_state[key] = load_history(username, domain)

    # Display the conversation so far
    for msg in st.session_state[key]:
        st.chat_message(msg["role"]).write(msg["content"])

    # Chat input box.  When the user submits text, we append it to
    # the in‑memory conversation and persist it to the database.  We
    # then call OpenAI's API with a system prompt and the chat
    # history to get the assistant's next message.
    if prompt := st.chat_input():
        # Append the user's message
        st.session_state[key].append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        save_msg(username, domain, "user", prompt)

        # Compose a minimal system prompt.  The assistant is asked
        # to pose seven concise questions about the selected domain.
        system = f"You are a Georgian data‑collector for {domain}. Ask 7 concise questions."
        messages = [{"role": "system", "content": system}] + st.session_state[key]

        # Call the OpenAI API.  The API key must be provided via
        # environment variable or .env file.  The model `gpt‑4o‑mini`
        # provides good quality at a reasonable cost.
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        try:
            reply = client.chat.completions.create(
                model="gpt-4o-mini", messages=messages
            ).choices[0].message.content
        except Exception as exc:
            # In case of an API error, log the error and inform the user
            reply = "სამწუხაროდ, AI‑თან დაკავშირება ვერ მოხერხდა."
            st.error(f"OpenAI error: {exc}")

        # Append assistant reply
        st.session_state[key].append({"role": "assistant", "content": reply})
        st.chat_message("assistant").write(reply)
        save_msg(username, domain, "assistant", reply)

    # A button to trigger fine‑tuning.  We only enable the button
    # when there is at least one message in the conversation.
    if st.button("🚀 დაიწყე ფაინ-ტუნინგი") and st.session_state[key]:
        st.info("ჩატის ისტორია შენახულია. ფაინ-ტუნინგი იწყება...")
        # Prepare environment variables for the subprocess.  We
        # propagate the current username and domain so that
        # `finetune.py` knows which chat history to use.
        env = os.environ.copy()
        env["CURRENT_USER"] = username
        env["CURRENT_DOMAIN"] = domain
        # Launch the fine‑tuning script.  Any errors will bubble up
        # and cause the page to display an exception.
        subprocess.run(["python", "finetune.py"], env=env, check=True)
        st.success("ფაინ-ტუნინგი დასრულდა! მოდელი შეგიძლიათ გამოიყენოთ.")

elif auth_status is False:
    # Display a message when authentication fails
    st.error("ავტორიზაცია ვერ მოხერხდა. სცადეთ ხელახლა.")
else:
    # Prompt the user to log in or register
    st.warning("გთხოვთ გაიაროთ ავტორიზაცია ან დარეგისტრირდით გვერდით.")