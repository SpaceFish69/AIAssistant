import streamlit as st
from whisper_listener import transcribe_live_audio
from llm_response import generate_response

st.title("🎤 AI Interview Assistant")
st.markdown("Listening for questions...")

response_placeholder = st.empty()

def on_transcription(text):
    response = generate_response(text)
    response_placeholder.markdown(f"**You Heard:** {text}\n\n**Suggested Answer:** {response}")

transcribe_live_audio(on_transcription)