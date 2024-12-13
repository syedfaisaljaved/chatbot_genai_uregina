import streamlit as st
from datetime import datetime
from pathlib import Path
from config import Config
from search_system import SearchSystem
import json
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatUI:
    def __init__(self):
        self.history_dir = Path("chat_history")
        self.history_dir.mkdir(exist_ok=True)
        self.initialize_session()
        self.typing_speed = 0.05

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
            # Add welcome message with markdown formatting
            welcome_message = {
                "role": "assistant",
                "content": """# Welcome to the University of Regina AI Assistant! 👋

I'm here to help you navigate information about our university. Here are some topics I can assist with:

* **Academic Programs**
  * Program requirements
  * Course information
  * Faculty-specific details

* **Admission Process**
  * Application requirements
  * Document submission
  * Important deadlines

* **Campus Life**
  * Student services
  * Housing options
  * Activities & events

* **Research & Resources**
  * Research opportunities
  * Library services
  * Lab facilities

Feel free to ask me anything about these topics or other aspects of university life at URegina!"""
            }
            st.session_state.messages.append(welcome_message)

        if 'search_system' not in st.session_state:
            logger.info("Initializing search system")
            config = Config()
            st.session_state.search_system = SearchSystem(config)

    def render(self):
        # Configure the page with dark theme
        st.set_page_config(
            page_title="URegina Assistant",
            page_icon="🎓",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Add custom CSS for dark theme and markdown
        st.markdown("""
        <style>
        /* Main theme colors and layout */
        .stApp {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #2D2D2D;
            border-right: 1px solid #3D3D3D;
        }

        /* Enhanced markdown styling */
        .markdown-text-container {
            color: #FFFFFF;
            font-family: 'Inter', sans-serif;
        }

        .markdown-text-container h1, 
        .markdown-text-container h2, 
        .markdown-text-container h3 {
            color: #FFFFFF;
            margin-top: 1em;
            margin-bottom: 0.5em;
        }

        /* Chat interface styling */
        .stChatMessage {
            background-color: #2D2D2D;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
        }

        /* Input box styling */
        .stTextInput > div > div {
            background-color: #2D2D2D;
            color: #FFFFFF;
            border: 1px solid #3D3D3D;
        }

        /* Button styling */
        .stButton > button {
            background-color: #2D2D2D;
            color: #FFFFFF;
            border: 1px solid #3D3D3D;
            border-radius: 5px;
        }

        .stButton > button:hover {
            background-color: #3D3D3D;
            border: 1px solid #4D4D4D;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #2D2D2D;
            color: #FFFFFF;
        }

        /* Link styling */
        a {
            color: #00ADB5;
            text-decoration: none;
        }

        a:hover {
            text-decoration: underline;
        }

        /* Debug info styling */
        .stExpander {
            background-color: #2D2D2D;
            border-radius: 5px;
            margin-top: 1em;
        }
        </style>
        """, unsafe_allow_html=True)

        # Sidebar
        with st.sidebar:
            st.header("Chat History")

            if st.button("💾 Save Chat"):
                self.save_history()
                logger.info("Chat history saved")

            if st.button("🗑️ Clear History"):
                logger.info("Clearing chat history")
                st.session_state.messages = []
                st.experimental_rerun()

            st.markdown("---")

            # Debug information in sidebar
            with st.expander("🔧 Debug Info"):
                st.write("Session State Keys:", list(st.session_state.keys()))
                st.write("Number of Messages:", len(st.session_state.messages))
                st.write("Message Types:", [msg["role"] for msg in st.session_state.messages])

        # Main chat interface
        col1, col2, col3 = st.columns([1, 6, 1])
        with col2:
            st.title("🎓 University of Regina AI Assistant")
            st.markdown("---")

            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "sources" in message and message["sources"]:
                        with st.expander("📚 Sources"):
                            for src in message["sources"]:
                                st.markdown(f"[{src['title']}]({src['url']})")

            # Chat input and response handling
            if prompt := st.chat_input("Ask about URegina..."):
                logger.info(f"New user input received: {prompt}")

                # Show user message
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Add user message to history
                st.session_state.messages.append({"role": "user", "content": prompt})

                # Show bot response
                with st.chat_message("assistant"):
                    try:
                        # Get response from search system
                        logger.info("Getting search response")
                        response = st.session_state.search_system.search(prompt)
                        logger.info("Search response received")

                        # Format the response with markdown
                        formatted_response = f"""### Response
{response['answer']}"""

                        # Update message with formatted response
                        st.markdown(formatted_response)

                        if response['sources']:
                            with st.expander("📚 Sources"):
                                for src in response['sources']:
                                    st.markdown(f"[{src['title']}]({src['url']})")

                        # Add assistant message to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": formatted_response,
                            "sources": response['sources']
                        })
                        logger.info("Assistant response added to chat history")

                    except Exception as e:
                        logger.error(f"Error processing response: {str(e)}")
                        st.error(f"Error: {str(e)}")

            # Debug chat history
            with st.expander("🔍 Debug Chat History"):
                st.json(st.session_state.messages)


def main():
    chat_ui = ChatUI()
    chat_ui.render()


if __name__ == "__main__":
    main()