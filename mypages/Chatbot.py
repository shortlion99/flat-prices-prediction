# mypages/Chatbot.py
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from chatbot.rag_chatbot import answer as rag_answer  # backend adapter

# helper lives at module scope (fine either here or inside show())
def set_new_message(message: str):
    """Flag a new message for processing on next rerun"""
    st.session_state.new_message = message
    st.session_state.chatbox = ""  # clear input

def show():
    st.title("💬 Property Price Chatbot")
    st.write("Ask me anything about Singapore's HDB resale market!")

    # Ensure .env is loaded when running Streamlit from repo root
    ROOT = Path(__file__).resolve().parents[1]
    load_dotenv(ROOT / ".env")

    # --- Init history & state ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "rag_state" not in st.session_state:
        # LangGraph-style state; your backend adapter mutates this
        st.session_state.rag_state = {"messages": []}
    if "new_message" not in st.session_state:
        st.session_state.new_message = None

    # --- Handle pending message (send to backend) ---
    if st.session_state.new_message:
        user_text = st.session_state.new_message
        st.session_state.chat_history.append({"role": "user", "text": user_text})

        with st.spinner("Thinking..."):
            try:
                # IMPORTANT: pass the shared rag_state so memory persists
                bot_text = rag_answer(
                    user_text,
                    conversation_state=st.session_state.rag_state
                )
            except Exception as e:
                bot_text = f"Sorry, I hit an error: {e}"

        st.session_state.chat_history.append({"role": "assistant", "text": bot_text})
        st.session_state.new_message = None  # reset

    # --- CSS for chat container + bubbles ---
    st.markdown(
        """
        <style>
        .block-container {padding-bottom: 1rem !important;}
        # in your st.markdown CSS block, REPLACE the .chat-container rule with:
        .chat-container {
            height: clamp(280px, 55vh, 75vh);  /* <— dynamic height */
            overflow-y: auto;
            padding: 1rem;
            padding-bottom: 2.5rem;
            border: 1px solid #ddd;
            border-radius: 10px;
            background-color: #f9f9f9;
            margin-bottom: 1rem;
        }
        .user-msg {
            background-color: #950606;
            padding: 8px 12px;
            border-radius: 10px;
            margin: 5px 0;
            text-align: right;
            color: white;
        }
        .bot-msg {
            padding: 8px 12px;
            border-radius: 10px;
            margin: 5px 0;
            text-align: left;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # --- Build chat messages inside container ---
    chat_html = "<div class='chat-container' id='chat-container'>"
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            chat_html += f"<div class='user-msg'>{msg['text']}</div>"
        else:
            chat_html += f"<div class='bot-msg'>{msg['text']}</div>"
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    # --- Auto-scroll chat container + autofocus ---
    st.markdown(
        """
        <script>
        function scrollChat() {
            const div = document.getElementById("chat-container");
            if (div) { div.scrollTop = div.scrollHeight; }
            const input = window.parent.document.querySelector('input[type="text"]');
            if (input) { input.focus(); }
        }
        setTimeout(scrollChat, 50);
        setTimeout(scrollChat, 150);
        setTimeout(scrollChat, 300);
        </script>
        """,
        unsafe_allow_html=True,
    )

    # --- Input area ---
    cols = st.columns([6, 1])
    with cols[0]:
        st.text_input(
            "Type your message...",
            key="chatbox",
            label_visibility="collapsed",
            on_change=lambda: set_new_message(st.session_state.chatbox),
        )
    with cols[1]:
        if st.button("➤", use_container_width=True) and st.session_state.get("chatbox"):
            set_new_message(st.session_state.chatbox)
