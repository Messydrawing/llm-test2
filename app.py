import streamlit as st

from data.dataset import build_dataset
from models.train import train_and_evaluate, TrainingConfig

st.title("LLM Distillation Demo")

if "trained" not in st.session_state:
    st.session_state.trained = False

if st.button("Generate Data and Train"):
    with st.spinner("Training model..."):
        train_data, val_data = build_dataset(4, 2, seq_len=20, seed=42)
        cfg = TrainingConfig()
        rate = train_and_evaluate(train_data, val_data, cfg)
    st.session_state.trained = True
    st.session_state.rate = rate

if st.session_state.trained:
    st.success(
        f"Training finished. Validation JSON success rate: {st.session_state.rate:.2%}"
    )
