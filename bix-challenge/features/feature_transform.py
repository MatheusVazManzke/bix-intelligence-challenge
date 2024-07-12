from transformer_classes import (
    RandomNoiseColumnsTransformer,
    DropNATransformer,
    KeepColumnsTransformer,
    TypeFloatTransformer,
    DataFrameSimpleImputer,
    DataFrameScaler,
    load_data,
    save_data,
)

from sklearn.pipeline import Pipeline


import os
import sys


important_columns = [
    "cn_001",
    "ca_000",
    "dt_000",
    "az_000",
    "cb_000",
    "bb_000",
    "du_000",
    "bu_000",
    "ay_003",
    "ag_006",
    "ai_000",
    "ba_000",
    "ba_004",
    "ee_003",
    "cn_006",
    "bf_000",
    "ag_003",
    "ba_008",
    "by_000",
    "ac_000",
    "cs_001",
    "cs_000",
    "cn_003",
    "ag_009",
    "ay_004",
    "cq_000",
    "cn_008",
    "bh_000",
    "cn_004",
    "ee_004",
    "cs_003",
    "dr_000",
    "ay_002",
    "dp_000",
    "ao_000",
    "ba_003",
    "cs_002",
    "cl_000",
    "ec_00",
    "bx_000",
    "az_002",
    "bc_000",
    "ay_006",
    "bs_000",
    "az_004",
    "cs_004",
    "az_001",
    "cj_000",
    "ah_000",
    "ba_005",
    "ay_000",
    "ap_000",
    "al_000",
    "an_000",
    "cc_000",
    "cn_002",
    "de_000",
    "aa_000",
    "bt_000",
    "ee_007",
    "aq_000",
    "bi_000",
    "am_0",
    "ee_005",
    "ag_002",
    "ay_008",
    "ay_005",
    "class",
]


def create_transformation_pipeline():
    return Pipeline(
        steps=[
            ("transform_dtypes_into_float", TypeFloatTransformer()),
            ("keep_important_columns", KeepColumnsTransformer(important_columns)),
            ("drop_columns_with_too_many_na", DropNATransformer(threshold=0.8)),
            ("imputer", DataFrameSimpleImputer(strategy="mean")),  # Impute NaNs with mean
            #  (
            #      "scaler",
            #      DataFrameScaler(),
            #  ),  # Standardize features by removing the mean and scaling to unit variance
        ]
    )


def main(raw_data_file):
    """Main function to execute the data transformation pipeline."""
    # Get the absolute path to the directory containing this script
    script_dir = os.path.dirname(os.path.abspath("feature_transform.py"))

    # Navigate to the project root directory (assuming it is two levels up)
    base_dir = os.path.abspath(os.path.join(script_dir, "../../"))

    # Construct the absolute path to the data file
    data_file_path = os.path.join(base_dir, f"data/{raw_data_file}")

    data = load_data(data_file_path)

    # Transform data
    pipeline = create_transformation_pipeline()
    treated_test_file = pipeline.fit_transform(data)

    # Save transformed data
    save_data(
        treated_test_file,
        os.path.abspath(
            os.path.join(base_dir, "data/processed/treated_air_system_present_year.csv")
        ),
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <raw_data_file>")
        sys.exit(1)

    raw_data_file = sys.argv[1]
    main(raw_data_file)
