import streamlit as st
from datetime import datetime
from pathlib import Path
from config import Config
from search_system import SearchSystem
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        st.success(f"Chat history saved to {history_file}")
        logger.info(f"Saved chat history to {history_file}")

    def load_history(self, file_path):
        with open(file_path, 'r') as f:
            st.session_state.messages = json.load(f)
        logger.info(f"Loaded chat history from {file_path}")
        st.success("Chat history loaded successfully")

    def initialize_session(self):
        logger.info("Current session state keys: %s", st.session_state.keys())

        if 'messages' not in st.session_state:
            logger.info("Initializing messages in session state")
            st.session_state.messages = []

        if 'search_system' not in st.session_state:
            logger.info("Initializing search system")
            config = Config()
            st.session_state.search_system = SearchSystem(config)

    def render(self):
        st.set_page_config(page_title="URegina Assistant", page_icon="🎓", layout="wide")

        with st.sidebar:
            st.header("Chat History")

            if st.button("Save Chat"):
                self.save_history()
                logger.info("Chat history saved")

            if st.button("Clear History"):
                logger.info("Clearing chat history")
                st.session_state.messages = []
                st.experimental_rerun()

            # Debug information in sidebar
            with st.expander("Debug Info"):
                st.write("Session State Keys:", list(st.session_state.keys()))
                st.write("Number of Messages:", len(st.session_state.messages))
                st.write("Message Types:", [msg["role"] for msg in st.session_state.messages])

        st.header("University of Regina AI Assistant")

        # Display existing chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("Sources"):
                        for src in message["sources"]:
                            st.markdown(f"[{src['title']}]({src['url']})")

        # Chat input and immediate response handling
        if prompt := st.chat_input("Ask about URegina..."):
            logger.info(f"New user input received: {prompt}")

            # Immediately show user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Show bot typing and get response
            with st.chat_message("assistant"):
                with st.empty():
                    st.markdown("Typing...")

                    try:
                        # Get response from search system
                        logger.info("Getting search response")
                        response = st.session_state.search_system.search(prompt)
                        logger.info("Search response received")

                        # Update message with response
                        st.markdown(response['answer'])

                        if response['sources']:
                            with st.expander("Sources"):
                                for src in response['sources']:
                                    st.markdown(f"[{src['title']}]({src['url']})")

                        # Add assistant message to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response['answer'],
                            "sources": response['sources']
                        })
                        logger.info("Assistant response added to chat history")

                    except Exception as e:
                        logger.error(f"Error processing response: {str(e)}")
                        st.error(f"Error: {str(e)}")

        # Debug chat history at bottom of page
        with st.expander("Debug Chat History"):
            st.json(st.session_state.messages)


def main():
    chat_ui = ChatUI()
    chat_ui.render()


if __name__ == "__main__":
    main()