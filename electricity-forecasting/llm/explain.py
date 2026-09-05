"""
Optional local LLM explanation assistant using Ollama.

Install Ollama separately and pull a supported local model before enabling:
    ollama pull phi3.5

This module is not required for the MVP.
"""

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def explain_forecast(summary, question, model="phi3.5"):
    prompt = f"""
You are an electricity forecasting assistant for a college project.

Forecast summary:
{summary}

User question:
{question}

Explain the result in simple, factual language. Do not invent data.
"""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("response", "")
