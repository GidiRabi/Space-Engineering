python3 -m venv venv && \
source venv/bin/activate && \
pip install opencv-python && \
mkdir -p .vscode && \
echo '{ "python.pythonPath": "venv/bin/python" }' > .vscode/settings.json
