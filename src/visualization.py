
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, TypedDict


class Config(TypedDict):
    name: str  # Label that will be displayed in graphs
    values: Dict[str, List[float | int]]

# Example
# configs = [
#     {
#         "name": "Dissimilarity",
#         "values": {
#             "Sick": np.random.normal(loc=0.5, scale=0.1, size=100),
#             "Normal": np.random.normal(loc=0.3, scale=0.1, size=100)
#         }
#     },
#     {
#         "name": "Energy",
#         "values": {
#             "Sick": np.random.normal(loc=0.7, scale=0.1, size=100),
#             "Normal": np.random.normal(loc=0.6, scale=0.1, size=100),
#             "Test": np.random.normal(loc=0.5, scale=0.1, size=100)
#         }
#     }
# ]


def plot_metric_distributions(configs: List[Config], output_path: str):
    num_metrics = len(configs)
    fig, axs = plt.subplots(num_metrics, 2, figsize=(12, num_metrics * 5))

    for i, config in enumerate(configs):
        metric_name = config['name']

        boxplot_data = []
        boxplot_labels = []
        for label, value in config['values'].items():
            axs[i, 0].hist(value, bins=20, alpha=0.5, label=label)

            boxplot_data.append(value)
            boxplot_labels.append(label)

        axs[i, 0].set_title(metric_name)
        axs[i, 0].legend()

        axs[i, 1].boxplot(boxplot_data, labels=boxplot_labels)
        axs[i, 1].set_title(metric_name)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
