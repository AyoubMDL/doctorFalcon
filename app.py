import streamlit as st
from streamlit_chat import message

counter = 0

st.title("💬 Doctor Falcon")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]



with st.form("chat_input", clear_on_submit=True):
    a, b = st.columns([4, 1])
    user_input = a.text_input(
        label="Your message:",
        placeholder="What would you like to say?",
        label_visibility="collapsed",
    )
    b.form_submit_button("Send", use_container_width=True)

for msg in st.session_state["messages"]:
    counter += 1
    message(msg["content"], is_user=msg["role"] == "user", key=str(counter))


if user_input:
    st.session_state['messages'].append({"role": "user", "content": user_input})
    counter += 1
    message(user_input, is_user=True, key=str(counter))
    msg = "i cant help you"
    st.session_state['messages'].append({"role": "assistant", "content": msg})
    counter += 1
    message(msg, key=str(counter))