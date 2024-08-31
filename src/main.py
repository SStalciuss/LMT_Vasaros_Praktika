from typing import List
import numpy as np
from scipy import stats
import pandas as pd
from glcm import calculate_glcm_for_multiple_images
from lbp import extract_lbp_features_from_images
from preprocess import (
    get_patient_list,
    circles_root_path,
    process_images_by_patient,
    process_all_images_by_patient,
)
from visualization import Config, plot_metric_distributions
from patients import Patients
from glrlm import calculate_glrlm_for_multiple_images
from fractal import calculate_fractal_features
from fourier_transform import calculate_fourier_features
from wavelet import calculate_wavelet_features

patients = Patients()
patients.load()
for patient in get_patient_list(circles_root_path):

    images = process_images_by_patient(patient, 6, 21)

    feature_providers = {
        "GLCM": lambda: calculate_glcm_for_multiple_images(images),
        "GLRLM": lambda: calculate_glrlm_for_multiple_images(images),
        "LBP": lambda: extract_lbp_features_from_images(images),
        "Fractal": lambda: calculate_fractal_features(images),
        "Fourier": lambda: calculate_fourier_features(images),
        "Wavelet": lambda: calculate_wavelet_features(images),
    }

    for provider_name, feature_provider in feature_providers.items():
        for feature_key, feature_value in feature_provider().items():
            patients.add_patient_feature(
                patient, f"{provider_name}_{feature_key}", feature_value
            )

patients.write()

configs: List[Config] = [
    {
        "name": feature,
        "values": {
            "A": patients.get_feture_for_class("A", feature),
            "B": patients.get_feture_for_class("B", feature),
        },
    }
    for feature in patients.get_all_features()
]
plot_metric_distributions(configs, "data/visualizations/test6.png")

results = []

for config in configs:
    feature_name = config["name"]
    values_A = config["values"]["A"]
    values_B = config["values"]["B"]

    statistic, p_value = stats.wilcoxon(values_A, values_B)
    mean_A = np.mean(values_A)
    mean_B = np.mean(values_B)
    variance_A = np.var(values_A)
    variance_B = np.var(values_B)

    results.append(
        {
            "Feature": feature_name,
            "p_value": p_value,
            "mean_A": mean_A,
            "mean_B": mean_B,
            "variance_A": variance_A,
            "variance_B": variance_B,
        }
    )

results_df = pd.DataFrame(results)
results_df.to_csv("wilcoxon_test_results.csv", index=False)

# from itertools import combinations

# def generate_combinations(available_features: list[str], max_feature_set_size: int):
#     for size in range(1, max_feature_set_size + 1):
#         for combo in combinations(available_features, size):
#             yield list(combo)

# # Example usage
# available_features = ["test", "test2", "test3", "test4", "test5", "test6", "test7", "test8", "test9", "test10"]
# max_feature_set_size = 5

# for combo in generate_combinations(available_features, max_feature_set_size):
#     print(combo)
