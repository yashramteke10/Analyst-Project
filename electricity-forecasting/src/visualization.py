import matplotlib.pyplot as plt


def plot_actual_vs_predicted(actual, predicted, index=None, title="Actual vs Predicted Electricity Consumption"):
    fig, ax = plt.subplots(figsize=(11, 4))
    x = index if index is not None else range(len(actual))
    ax.plot(x, actual, label="Actual", linewidth=1.5)
    ax.plot(x, predicted, label="Predicted", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Consumption (kWh)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig
