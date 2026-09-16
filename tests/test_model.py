from yakupredict_ai.data_generator import generate_dataset
from yakupredict_ai.model import train_bundle


def test_training_and_prediction_stays_in_valid_range():
    df = generate_dataset(rows_per_scenario=700, seed=7)
    bundle, report, comparison = train_bundle(df)
    sample = df[df.scenario_id == 9].iloc[-8:].copy()
    scores = bundle.predict_scores(sample)
    assert len(scores) == 8
    assert scores["risk_score"].between(0, 1).all()
    assert set(scores["prediction"].unique()).issubset({0, 1})
    assert 0 <= report["test"]["f1"] <= 1
    assert set(comparison["model"]) == {"Random Forest", "Isolation Forest", "Híbrido"}
    assert abs(bundle.supervised_weight + bundle.anomaly_weight - 1.0) < 1e-9
