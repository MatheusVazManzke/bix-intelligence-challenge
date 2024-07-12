from transformer_classes import (
    RandomNoiseColumnsTransformer,
    DropNATransformers,
    KeepColumnsTransformer,
    load_data,
    save_data,
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


import os
import sys


important_columns = [
    "bx_000",
    "az_007",
    "cs_005",
    "cn_002",
    "ee_009",
    "bs_000",
    "bv_000",
    "dr_000",
    "ca_000",
    "by_000",
    "ba_000",
    "am_0",
    "cs_006",
    "ba_006",
    "al_000",
    "cs_001",
    "cc_000",
    "bt_000",
    "ee_005",
    "cj_000",
    "ay_002",
    "ag_007",
    "ac_000",
    "aa_000",
    "ay_007",
    "ai_000",
    "ay_000",
    "ay_008",
    "ag_002",
    "ay_005",
]


def create_transformation_pipeline():
    return Pipeline(
        steps=[
            ("keep_important_columns", KeepColumnsTransformer(important_columns)),
            ("add_random_noise_column"),
            RandomNoiseColumnsTransformer(),
            ("drop_columns_with_too_many_na", DropNATransformer(threshold=0.8))(
                "imputer", SimpleImputer(strategy="mean")
            ),  # Impute NaNs with mean
            (
                "scaler",
                StandardScaler(),
            ),  # Standardize features by removing the mean and scaling to unit variance
        ]
    )


def main(raw_data_file):
    """Main function to execute the data transformation pipeline."""
    # Get the absolute path to the directory containing this script
    script_dir = os.path.dirname(os.path.abspath("create_train_dataset.py"))

    # Navigate to the project root directory (assuming it is two levels up)
    base_dir = os.path.abspath(os.path.join(script_dir, "../../"))

    # Construct the absolute path to the data file
    data_file_path = os.path.join(base_dir, f"data/processed/{raw_data_file}")

    data = load_data(data_file_path)

    # Transform data
    pipeline = Pipeline()

    # Save transformed data
    save_data(
        filtered_data,
        os.path.abspath(os.path.join(base_dir, "data/processed/train_dataset.csv")),
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <raw_data_file>")
        sys.exit(1)

    raw_data_file = sys.argv[1]
    main(raw_data_file)
