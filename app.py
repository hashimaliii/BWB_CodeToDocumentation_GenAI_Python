import os
import json
import streamlit as st
import pandas as pd

import torch
from docgen_system import load_bpe_models, load_word2vec_model, load_bilstm, generate_for_function

st.set_page_config(page_title="Documentation Generator", layout="wide")

@st.cache_resource(show_spinner=False)
def load_components(bilstm_ckpt: str, word2vec_path: str | None):
    bpe_models = load_bpe_models()
    w2v = load_word2vec_model(word2vec_path)
    bilstm = load_bilstm(bilstm_ckpt)
    return bpe_models, w2v, bilstm

st.title("Documentation Generation System")

with st.sidebar:
    st.header("Settings")
    bilstm_ckpt = st.text_input("BiLSTM checkpoint", value=os.path.join("bilstm", "bilstm_best.pt"))
    word2vec_path = st.text_input("Word2Vec model (optional)", value="")
    max_tokens = st.number_input("Max new tokens", min_value=20, max_value=400, value=120, step=10)
    bpe_choice = st.selectbox("BPE Model", options=["combined", "code", "doc"], index=0)
    run_load = st.button("Load Models")

if 'components' not in st.session_state and os.path.exists(os.path.join("bilstm", "bilstm_best.pt")):
    st.session_state['components'] = load_components(os.path.join("bilstm", "bilstm_best.pt"), None)

if run_load:
    w2v_arg = word2vec_path if word2vec_path.strip() else None
    st.session_state['components'] = load_components(bilstm_ckpt, w2v_arg)

if 'components' not in st.session_state:
    st.info("Load models from the sidebar to begin.")
else:
    bpe_models, w2v, bilstm = st.session_state['components']

    tabs = st.tabs(["Single Input", "Batch CSV"])

    with tabs[0]:
        st.subheader("Single Function/Code Input")
        code = st.text_area("Paste function code here", height=220, placeholder="def add(a,b):\n    return a+b")
        name = st.text_input("Function name (optional)", value="")
        if st.button("Generate Documentation", type="primary"):
            if not code.strip():
                st.warning("Please paste code.")
            else:
                with st.spinner("Generating..."):
                    res = generate_for_function(code, name if name.strip() else None, bilstm, bpe_models, w2v, int(max_tokens), bpe_kind=bpe_choice)
                st.success("Done")
                st.write("### Summary")
                st.write(res["summary"]) 
                st.write("### Docstring")
                st.code(res["docstring"], language="markdown")

    with tabs[1]:
        st.subheader("Batch Generation from CSV")
        csv_file = st.file_uploader("Upload CSV with columns: code, function_name", type=["csv"])
        sample_size = st.number_input("Sample size (0 for all)", min_value=0, value=100, step=50)
        if st.button("Generate Batch"):
            if not csv_file:
                st.warning("Please upload a CSV file.")
            else:
                df = pd.read_csv(csv_file)
                if sample_size > 0 and len(df) > sample_size:
                    df = df.head(sample_size)
                outputs = []
                with st.spinner("Generating batch..."):
                    for _, row in df.iterrows():
                        code = str(row.get("code", ""))
                        name = row.get("function_name")
                        res = generate_for_function(code, name, bilstm, bpe_models, w2v, int(max_tokens), bpe_kind=bpe_choice)
                        outputs.append({"name": name, "summary": res["summary"], "docstring": res["docstring"]})
                st.success(f"Generated {len(outputs)} items")
                out_json = json.dumps(outputs, indent=2, ensure_ascii=False)
                st.download_button("Download JSON", data=out_json.encode('utf-8'), file_name="generated_docs.json", mime="application/json")
