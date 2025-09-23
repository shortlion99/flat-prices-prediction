import streamlit as st

def show():
    st.title("💬 Property Price Chatbot")
    st.write("Ask me anything about Singapore's HDB resale market!")

    # --- Init history & state ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "new_message" not in st.session_state:
        st.session_state.new_message = None

    # --- Handle pending message ---
    if st.session_state.new_message:
        st.session_state.chat_history.append({"role": "user", "text": st.session_state.new_message})
        st.session_state.chat_history.append({"role": "assistant", "text": "test"})
        st.session_state.new_message = None  # reset

    # --- CSS for chat container + bubbles ---
    st.markdown(
        """
        <style>
        .block-container {padding-bottom: 1rem !important;}
        .chat-container {
            min-height: 300px;
            max-height: 300px;
            overflow-y: auto;
            padding: 1rem;
            padding-bottom: 2.5rem; /* extra breathing space */
            border: 1px solid #ddd;
            border-radius: 10px;
            background-color: #f9f9f9;
            margin-bottom: 1rem;
        }
        .user-msg {
            background-color: #FEC9D1;
            padding: 8px 12px;
            border-radius: 10px;
            margin: 5px 0;
            text-align: right;
        }
        .bot-msg {
            background-color: #E5E5EA;
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
    chat_html += "</div>"  # no anchor needed anymore
    st.markdown(chat_html, unsafe_allow_html=True)

    # --- Auto-scroll chat container + autofocus ---
    st.markdown(
        """
        <script>
        function scrollChat() {
            const div = document.getElementById("chat-container");
            if (div) {
                div.scrollTop = div.scrollHeight; // force scroll to bottom
            }
            const input = window.parent.document.querySelector('input[type="text"]');
            if (input) { input.focus(); }
        }
        // Run multiple times to catch DOM refresh
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


def set_new_message(message: str):
    """Flag a new message for processing on next rerun"""
    st.session_state.new_message = message
    st.session_state.chatbox = ""  # clear input
