from yakupredict_ai.data_generator import generate_dataset
from yakupredict_ai.features import engineer_features
from yakupredict_ai.config import MODEL_FEATURES


def test_engineered_features_are_created():
    df = generate_dataset(rows_per_scenario=40, seed=10)
    features = engineer_features(df)
    assert all(column in features.columns for column in MODEL_FEATURES)
    assert features[MODEL_FEATURES].shape[0] == len(df)
