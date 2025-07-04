#!/bin/bash

# Define the tmux session name
SESSION_NAME="inference_env"

# Define your conda environment name
CONDA_ENV="inference"

# Define your application paths
MAIN_APP="app/main.py"
UI_APP="app/ui.py"

# Check if the tmux session already exists
tmux has-session -t "${SESSION_NAME}" 2>/dev/null

if [ $? != 0 ]; then
  echo "Creating new tmux session: ${SESSION_NAME}"

  # Create a new tmux session with the first window for main.py
  # -d detaches the session so the script can continue
  tmux new-session -s "${SESSION_NAME}" -n "Backend (main.py)" -d

  # Send commands to the first window (Backend)
  # Activate conda environment
  tmux send-keys -t "${SESSION_NAME}:0" "conda activate ${CONDA_ENV}" C-m
  # Run the main Python application
  tmux send-keys -t "${SESSION_NAME}:0" "python3 ${MAIN_APP}" C-m

  # Create a new window for the Streamlit UI
  tmux new-window -t "${SESSION_NAME}" -n "Streamlit UI (ui.py)"

  # Send commands to the second window (Streamlit UI)
  # Activate conda environment
  tmux send-keys -t "${SESSION_NAME}:1" "conda activate ${CONDA_ENV}" C-m
  # Run the Streamlit application
  tmux send-keys -t "${SESSION_NAME}:1" "streamlit run ${UI_APP}" C-m

  echo "Tmux session '${SESSION_NAME}' created with two windows."
  echo "Window 0: Backend (${MAIN_APP})"
  echo "Window 1: Streamlit UI (${UI_APP})"
  echo ""
  echo "To attach to the session, run: tmux attach-session -t ${SESSION_NAME}"
  echo "Once attached, you can switch between windows using Ctrl+b, then n (next) or p (previous), or by number (e.g., Ctrl+b, 0 or Ctrl+b, 1)."
  echo "The Streamlit app will usually be accessible at http://localhost:8501 (check the output in the Streamlit UI window)."
else
  echo "Tmux session '${SESSION_NAME}' already exists. Attaching to it."
  tmux attach-session -t "${SESSION_NAME}"
fi
