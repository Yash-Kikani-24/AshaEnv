# server/ — FastAPI HTTP wrapper around AshaEnv.
#
# Exposes the environment as a REST API so any external agent (LLM, script,
# or evaluation harness) can interact with it over HTTP instead of importing
# Python directly.
#
# app.py   : defines all endpoints (/reset, /step, /state, /health)
#
# To run the server locally:
#   uvicorn server.app:app --host 0.0.0.0 --port 7860
#
# The Dockerfile at the project root builds and starts this server automatically.
