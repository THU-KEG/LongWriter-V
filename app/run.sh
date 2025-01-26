#!/bin/bash

# Add the project root to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

cd app

# Kill any existing Streamlit processes
pkill -f "streamlit run"

# Create logs directory with timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")
mkdir -p "logs/${timestamp}"

# Start the unified application
nohup streamlit run main.py --server.port 9817 --server.address 0.0.0.0 > "logs/${timestamp}/app.log" 2>&1 &

echo "Application started. Access via:"
echo "- Annotation Platform: http://localhost:9817/annotation"
echo "- Script Generation: http://localhost:9817/show_script"
echo "- Admin Backend: http://localhost:9817/admin"
echo "Check logs at logs/${timestamp}/app.log" 