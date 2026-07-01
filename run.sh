#!/usr/bin/env bash
# Start the Streamlit app
cd "$(dirname "$0")"
source .venv/bin/activate
streamlit run aus_econ_model/streamlit_app.py --server.runOnSave true "$@"
