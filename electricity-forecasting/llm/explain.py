"""Optional local-LLM explanation integration placeholder; Ollama is not required for MVP."""

def explain_metrics(metrics):
    return (f"MAE: {metrics['mae']:.3f}; RMSE: {metrics['rmse']:.3f}. "
            "Lower MAE and RMSE indicate predictions closer to actual consumption.")
