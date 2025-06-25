import streamlit as st
import sys
import types
import torch
import gc
import io
import random
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from diffusers import StableDiffusionPipeline
from gtts import gTTS
from PIL import Image

# Fix for torch path issue
if not isinstance(torch.__path__, list):
    torch.__path__ = [p for p in sys.path if "torch" in p or "site-packages" in p]
torch.classes = types.SimpleNamespace()

# Page config
st.set_page_config(page_title="MultiMind", page_icon="🧠", layout="wide")

# Theme toggle UI
st.markdown("""
<style>
.theme-toggle {position: fixed; top: 15px; right: 15px; background: #FFD700; color: black; border-radius: 10px; padding: 5px 10px; font-weight: bold; z-index: 9999;}
.light-mode {background-color: #fff !important; color: #000 !important;}
.light-mode h1, .light-mode .stButton > button, .light-mode .stDownloadButton > button {color: #000 !important;}
</style>
<script>
const toggle = document.createElement('button');
toggle.innerText = '🌓 Toggle Theme';
toggle.className = 'theme-toggle';
toggle.onclick = () => document.body.classList.toggle('light-mode');
document.body.appendChild(toggle);
</script>
""", unsafe_allow_html=True)

# Splash screen (first-time only)
if 'splash_shown' not in st.session_state:
    st.markdown("""
    <style>
    .splash {position: fixed; top: 0; left: 0; width:100%; height:100%; background:#000000cc; display:flex; align-items:center; justify-content:center; z-index:9999; flex-direction:column; animation: fadeOut 2s ease-in-out forwards 2s;}
    @keyframes fadeOut {to {opacity: 0; visibility: hidden;}}
    .splash h1 {font-size:3rem; color:#FFD700; text-shadow:2px 2px #000;}
    .close-btn {background:#FFD700; border:none; padding:10px 20px; font-size:1.2rem; font-weight:bold; border-radius:12px; cursor:pointer;}
    </style>
    <div class='splash'>
      <h1>Welcome to MultiMind 👋<br>Your All-in-One AI Tool</h1>
    </div>
    """, unsafe_allow_html=True)
    st.session_state.splash_shown = True

# Load pipelines
@st.cache_resource
def get_pipeline(task):
    if task == "Summarization":
        model_name = "facebook/bart-large-cnn"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        def summarize(text):
            inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True, padding="longest").to(model.device)
            ids = model.generate(inputs["input_ids"], max_length=120, min_length=30, do_sample=False)
            return tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
        return summarize
    if task == "Next Word Prediction":
        return pipeline("text-generation", model="gpt2")
    if task == "Story Prediction":
        return pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")
    if task == "Chatbot":
        return pipeline("text-generation", model="microsoft/DialoGPT-medium")
    if task == "Sentiment Analysis":
        return pipeline("sentiment-analysis")
    if task == "Question Answering":
        return pipeline("question-answering", model="deepset/roberta-base-squad2")
    if task == "Image Generation":
        return StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None, requires_safety_checker=False
        )

# Task selector
task = st.selectbox("🎯 Choose Your Smart Tool", [
    "Summarization", "Next Word Prediction", "Story Prediction",
    "Chatbot", "Sentiment Analysis", "Question Answering", "Image Generation"
], key="task_selector")

st.title("🧠 MultiMind – AI Assistant")

# Initialize states
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "user_input" not in st.session_state: st.session_state.user_input = ""

# Task-specific intro and animation
intro_map = {
    "Summarization": ("📚 Summarize long texts into insights.", "3o6Zt481isNVuQI1l6"),
    "Next Word Prediction": ("🔮 Predict the next words.", "l41lFw057lAJQMwg0"),
    "Story Prediction": ("📖 Continue your story.", "13FrpeVH09Zrb2"),
    "Chatbot": ("💬 Chat freely with AI.", "xUPGcguWZHRC2HyBRS"),
    "Sentiment Analysis": ("📊 Gauge message mood.", "TdfyKrN7HGTIY"),
    "Question Answering": ("❓ Ask from text.", "3oEjHGr1t1rS2yWQ2Y"),
    "Image Generation": ("🎨 Create AI images.", "l4FGGafcOHmrlQxG0")
}
intro, gif = intro_map[task]
st.markdown(f"<div style='text-align:center; color:#FFD700; font-size:1.1rem;'>{intro}</div>", unsafe_allow_html=True)
st.markdown(f"<div style='text-align:center;'><img src='https://media.giphy.com/media/{gif}/giphy.gif' width='300'></div>", unsafe_allow_html=True)

# Chatbot handler
if task == "Chatbot":
    cols = st.columns(3)
    sample_prompts = ["What's the weather today?", "Tell me a joke", "Who is Elon Musk?"]
    selected_prompt = None
    for col, prompt in zip(cols, sample_prompts):
        if col.button(prompt): selected_prompt = prompt

    user_input = selected_prompt if selected_prompt else st.text_area("Your message:", key="chat_input", height=200)

    if st.button("Send"):
        if user_input.strip():
            pipe = get_pipeline("Chatbot")
            raw = pipe(f"User: {user_input}\nBot:", max_length=150, temperature=0.7, top_k=40, top_p=0.9, do_sample=True, pad_token_id=50256)[0]["generated_text"]
            response = raw.split("Bot:")[-1].strip() if "Bot:" in raw else raw.strip()
            for tag in ["notwithstandingUser :", "analogousUser :", "User :", "User:"]:
                response = response.replace(tag, "").strip()
            if response.lower() == user_input.lower() or not response:
                response = random.choice([
                    "🤖 I'm still learning. Try rephrasing.",
                    "Can you say that differently?",
                    "Let me think... Ask another way."
                ])
            st.session_state.chat_history.append({"user": user_input, "bot": response})
            st.session_state.user_input = ""

# Chat history UI
if task == "Chatbot" and st.session_state.chat_history:
    st.markdown("---")
    for turn in st.session_state.chat_history:
        st.markdown(f"**You:** {turn['user']}")
        st.markdown(f"**Bot:** {turn['bot']}")
    if st.button("🧹 Clear Chat"): st.session_state.chat_history = []

# Common input for other tasks
if task != "Chatbot":
    user_input = st.text_area("Enter your text:", height=200, key="main_text")

# Text tasks runner
def run_text_task():
    pipe = get_pipeline(task)
    if task == "Summarization": return pipe(user_input)
    if task == "Next Word Prediction": return pipe(user_input, max_length=150, num_return_sequences=1, temperature=0.8, top_k=50, top_p=0.95, do_sample=True, pad_token_id=50256)[0]["generated_text"]
    if task == "Story Prediction": return pipe(user_input, max_length=200, num_return_sequences=1, temperature=0.9, top_k=50, top_p=0.95, do_sample=True, pad_token_id=50256)[0]["generated_text"]
    if task == "Sentiment Analysis":
        result = pipe(user_input)[0]
        return f"{result['label']} ({round(result['score']*100, 2)}%)"

# Question Answering
if task == "Question Answering":
    context = st.text_area("Context:", height=150)
    question = st.text_input("Your Question:")
    if st.button("Get Answer") and context and question:
        with st.spinner("Processing..."):
            answer = get_pipeline(task)({"question": question, "context": context})['answer']
            st.success(answer)
            st.download_button("Download Answer", answer, "answer.txt", "text/plain")

# Image Generation
if task == "Image Generation" and user_input:
    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            image = get_pipeline(task).to("cuda" if torch.cuda.is_available() else "cpu")(user_input).images[0]
            st.image(image, caption="Generated Image")
            buf = io.BytesIO(); image.save(buf, format="PNG")
            st.download_button("Download Image", buf.getvalue(), "image.png", "image/png")

# Run text tasks
if task in ["Summarization", "Next Word Prediction", "Story Prediction", "Sentiment Analysis"] and user_input:
    if st.button("Run Task"):
        with st.spinner("Processing..."):
            output = run_text_task()
            st.success(output)
            st.download_button("Download Result", output, "result.txt", "text/plain")
            if st.checkbox("🔊 Play Audio"):
                tts = gTTS(output, lang='en')
                tts.save("result.mp3")
                st.audio("result.mp3")
