import json
import numpy as np
from glcm import GlcmFeatures

type FeatureValue = int | float


class Patients:
    patient_data: dict

    def __init__(self) -> None:
        self.patient_data = {}

    def load(self) -> None:
        try:
            with open("patient_features.json", "r", encoding="utf-8") as json_file:
                self.patient_data = json.load(json_file)
        except FileNotFoundError:
            print("No existing patient data found. Starting with an empty dataset.")
        except json.JSONDecodeError as e:
            print(f"Error loading patient data: {e}")

    def write(self) -> None:
        with open("patient_features.json", "w", encoding="utf-8") as json_file:
            json.dump(self.patient_data, json_file, indent=4, sort_keys=True)

    def add_patient_feature(
        self, patient: str, feature: str, value: FeatureValue
    ) -> None:
        if patient not in self.patient_data:
            self.patient_data[patient] = {}
        self.patient_data[patient][feature] = self._convert_to_serializable(value)

    def _convert_to_serializable(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return obj

    def get_feture_for_class(
        self, class_id: str, feature: GlcmFeatures
    ) -> list[FeatureValue]:
        values = []
        for patient_key, features in self.patient_data.items():
            if class_id in patient_key:
                values.append(features[feature])

        return values

    def get_all_features(self) -> list[str]:
        if not self.patient_data:
            return []
        first_patient_features = next(iter(self.patient_data.values()))
        return list(first_patient_features.keys())
