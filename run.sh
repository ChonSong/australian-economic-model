#!/usr/bin/env bash
# Start the Streamlit app — bootstrap venv if missing
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi
source .venv/bin/activate
streamlit run aus_econ_model/streamlit_app.py --server.runOnSave true "$@"
