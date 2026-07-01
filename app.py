# 🚀 FINAL STREAMLIT CHATBOT (VOICE + TYPING + CONTEXT)

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import speech_recognition as sr

st.set_page_config(page_title="AI Chatbot", layout="wide")

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- SESSION ----------------
if "chats" not in st.session_state:
    st.session_state.chats = {"New Chat": []}

if "current_chat" not in st.session_state:
    st.session_state.current_chat = "New Chat"

if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None

# ---------------- SIDEBAR ----------------
st.sidebar.title("💬 Chats")

if st.sidebar.button("➕ New Chat"):
    name = f"Chat {len(st.session_state.chats)+1}"
    st.session_state.chats[name] = []
    st.session_state.current_chat = name
    st.session_state.chat_history_ids = None

for chat_name in st.session_state.chats.keys():
    if st.sidebar.button(chat_name):
        st.session_state.current_chat = chat_name
        st.session_state.chat_history_ids = None

# ---------------- EXPORT ----------------
def export_chat():
    chat = st.session_state.chats[st.session_state.current_chat]
    text = "\n".join([f"{m['role']}: {m['text']}" for m in chat])
    st.sidebar.download_button("📥 Download Chat", text, file_name="chat.txt")

export_chat()

# ---------------- CHAT AREA ----------------
st.title("🤖 Smart Chatbot")

chat = st.session_state.chats[st.session_state.current_chat]

for msg in chat:
    role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
    st.markdown(f"**{role}:** {msg['text']}")

# ---------------- VOICE FUNCTION ----------------
def get_voice_input():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening...")
            audio = r.listen(source, timeout=5)
        text = r.recognize_google(audio)
        return text
    except:
        return ""

# ---------------- INPUT ----------------
col1, col2, col3 = st.columns([6,1,1])

with col1:
    user_input = st.text_input("Type your message...")

with col2:
    send_clicked = st.button("Send")

with col3:
    mic_clicked = st.button("🎤")

# ---------------- RESPONSE ----------------
def stream_response(user_text):
    global_ids = st.session_state.chat_history_ids

    new_input_ids = tokenizer.encode(user_text + tokenizer.eos_token, return_tensors="pt")

    if global_ids is not None:
        bot_input_ids = torch.cat([global_ids, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids

    output_ids = model.generate(
        bot_input_ids,
        max_length=500,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        repetition_penalty=1.2,
        no_repeat_ngram_size=2
    )

    st.session_state.chat_history_ids = output_ids

    text = tokenizer.decode(
        output_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    ).strip()

    if text == "" or len(text) < 2:
        text = "Hey! 😊 How can I help you?"

    words = text.split(" ")

    for i in range(len(words)):
        yield " ".join(words[:i+1])
        time.sleep(0.03)

# ---------------- HANDLE SEND ----------------
def handle_message(input_text):
    chat.append({"role": "user", "text": input_text})

    placeholder = st.empty()
    response_text = ""

    # typing animation
    for i in range(3):
        placeholder.markdown(f"**🤖 Bot:** typing{'.'*i}")
        time.sleep(0.3)

    for chunk in stream_response(input_text):
        response_text = chunk
        placeholder.markdown(f"**🤖 Bot:** {response_text}")

    chat.append({"role": "assistant", "text": response_text})

# ---------------- BUTTON ACTIONS ----------------
if send_clicked and user_input:
    handle_message(user_input)

if mic_clicked:
    voice_text = get_voice_input()
    if voice_text:
        st.success(f"🎤 You said: {voice_text}")
        handle_message(voice_text)
    else:
        st.warning("Could not understand audio")

# ---------------- STYLE ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg,#0f172a,#1e293b);
}
</style>
""", unsafe_allow_html=True)