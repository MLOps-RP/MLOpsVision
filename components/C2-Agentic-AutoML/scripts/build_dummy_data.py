"""Create dummy C1 outputs for development (run once)."""
from c2_automl.io.dummy_c1 import build_from_openml

if __name__ == "__main__":
    for key in ["adult", "diabetes"]:
        c1 = build_from_openml(key)
        print(f"built {c1.dataset_id}: {c1.class_distribution}")

    # Same Adult data, but positive class is only 3% -> tests the imbalance diagnosis
    c1 = build_from_openml("adult", dataset_id="adult_imbalanced", minority_share=0.03)
    print(f"built {c1.dataset_id}: {c1.class_distribution}")
