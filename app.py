# app.py
import streamlit as st
from datetime import datetime
from pathlib import Path
from config import Config
from search_system import SearchSystem
import json


class ChatUI:
    def __init__(self):
        self.history_dir = Path("chat_history")
        self.history_dir.mkdir(exist_ok=True)
        self.initialize_session()

    def save_history(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = self.history_dir / f"chat_{timestamp}.json"
        with open(history_file, 'w') as f:
            json.dump(st.session_state.messages, f)

    def load_history(self, file_path):
        with open(file_path, 'r') as f:
            st.session_state.messages = json.load(f)

    def initialize_session(self):
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if not hasattr(st.session_state, 'search_system'):
            config = Config()
            st.session_state.search_system = SearchSystem(config)

    def render(self):
        st.set_page_config(page_title="URegina Assistant", page_icon="🎓", layout="wide")

        with st.sidebar:
            st.header("Chat History")
            if st.button("Save Chat"):
                self.save_history()

            history_files = list(self.history_dir.glob("*.json"))
            if history_files:
                selected_file = st.selectbox(
                    "Load Previous Chat",
                    history_files,
                    format_func=lambda x: x.stem
                )
                if st.button("Load"):
                    self.load_history(selected_file)
                if st.button("Clear History"):
                    st.session_state.messages = []

        st.header("University of Regina AI Assistant")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander("Sources"):
                        for src in msg["sources"]:
                            st.write(f"[{src['title']}]({src['url']})")

        if prompt := st.chat_input("Ask about URegina..."):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.search_system.search(prompt)
                    st.write(response['answer'])

                    if response['sources']:
                        with st.expander("Sources"):
                            for src in response['sources']:
                                st.write(f"[{src['title']}]({src['url']})")

            st.session_state.messages.append({
                "role": "assistant",
                "content": response['answer'],
                "sources": response['sources']
            })

