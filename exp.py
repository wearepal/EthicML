"""Experiments on ProPublica."""

from dataclasses import dataclass
from typing_extensions import Self, override

import numpy as np
from sklearn.linear_model import LogisticRegression

from ethicml import (
    DataTuple,
    Prediction,
    Results,
    SoftPrediction,
    TestTuple,
    data,
    metrics,
    models,
    run,
)

# %%


@dataclass
class Hybrid(models.InAlgorithmNoParams):
    """Hybrid algorithm."""

    @property
    @override
    def name(self) -> str:
        return "Hybrid"

    @override
    def fit(self, train: DataTuple, seed: int = 888) -> Self:
        if train.name is None or "Compas" not in train.name:
            raise RuntimeError("The Hybrid algorithm only works on the COMPAS dataset")
        random_state = np.random.RandomState(seed=seed)
        sex = train.s.to_numpy() if train.s.name == "sex" else train.x["sex"].to_numpy()
        male = 1
        priors_age = train.x[["priors-count", "age-num"]].to_numpy()
        priors_age_male = priors_age[sex == male]
        y_male = train.y.to_numpy()[sex == male].ravel()
        self.clf_male = LogisticRegression(
            solver="liblinear", random_state=random_state, multi_class="auto"
        )
        self.clf_male.fit(priors_age_male, y_male)
        priors_age_female = priors_age[sex != male]
        y_female = train.y.to_numpy()[sex != male].ravel()
        self.clf_female = LogisticRegression(
            solver="liblinear", random_state=random_state, multi_class="auto"
        )
        self.clf_female.fit(priors_age_female, y_female)
        return self

    @override
    def predict(self, test: TestTuple) -> Prediction:
        if test.name is None or "Compas" not in test.name:
            raise RuntimeError("The Hybrid algorithm only works on the COMPAS dataset")
        sex = test.s.to_numpy() if test.s.name == "sex" else test.x["sex"].to_numpy()
        male = 1
        priors_age = test.x[["priors-count", "age-num"]].to_numpy()
        pred = np.where(
            (sex == male)[:, np.newaxis],
            self.clf_male.predict_proba(priors_age),
            self.clf_female.predict_proba(priors_age),
        )
        return SoftPrediction(soft=pred)

    @override
    def run(self, train: DataTuple, test: TestTuple, seed: int = 888) -> Prediction:
        return self.fit(train, seed=seed).predict(test)


def _run() -> Results:
    results = run.evaluate_models(
        [data.Compas(split=data.CompasSplits.RACE)],
        inprocess_models=[models.Corels(), models.LR(), Hybrid()],
        # inprocess_models=[Hybrid()],
        metrics=[metrics.Accuracy(), metrics.TPR()],
        per_sens_metrics=[metrics.Accuracy(), metrics.ProbPos(), metrics.TPR()],
        num_jobs=1,
    )
    return results


r = _run()
r["Accuracy"]
r["Accuracy_race_0÷race_1"]
r["TPR_race_0÷race_1"]
