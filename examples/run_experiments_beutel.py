import os
import sys
from pathlib import Path

import pandas as pd

from ethicml.algorithms.inprocess import LR, SVM, Agarwal, Kamishima
from ethicml.algorithms.inprocess.kamiran import Kamiran
from ethicml.data.load import create_data_obj, load_data
from ethicml.data.tabular_data.dataset import Dataset
from ethicml.evaluators.cross_validator import CrossValidator
from ethicml.evaluators.evaluate_models import run_metrics
from ethicml.metrics import CV, TPR, Accuracy, ProbPos
from ethicml.utility.data_structures import DataTuple

metrics = [Accuracy(), CV()]
per_sens_metrics = [Accuracy(), TPR(), ProbPos()]


def run_experiments(train, test, exp_name, seed):
    svm_hyperparams = {"C": [10 ** x for x in range(0, 6)], "kernel": ["linear"]}
    svm_cv = CrossValidator(SVM, svm_hyperparams, folds=3)
    svm_cv.run(train)
    best_svm = svm_cv.best(Accuracy())
    best_svm_hyperparams = svm_cv.best_hyper_params(Accuracy())
    print(f"seed_{seed}_svm_CV_completed")

    lr_hyperparams = {"C": [10 ** x for x in range(-4, 4)]}
    lr_cv = CrossValidator(LR, lr_hyperparams, folds=3)
    lr_cv.run(train)
    best_lr = lr_cv.best(Accuracy())
    best_lr_hyperparams = lr_cv.best_hyper_params(Accuracy())
    print(f"seed_{seed}_lr_CV_completed")

    models = [
        best_svm,
        best_lr,
        Kamishima(),
        Agarwal(**best_lr_hyperparams, fairness="EqOd", classifier="LR"),
        Agarwal(**best_svm_hyperparams, fairness="EqOd", classifier="SVM"),
        Kamiran(**best_lr_hyperparams, classifier="LR"),
        Kamiran(**best_svm_hyperparams, classifier="SVM"),
    ]

    columns = ["dataset", "transform", "model", "repeat"]
    columns += [metric.name for metric in metrics]
    results = pd.DataFrame(columns=columns)

    for model in models:
        temp_res = {
            "dataset": "Adult",
            "transform": "beutel",
            "model": model.name,
            "repeat": f"{seed}",
        }

        predictions: pd.DataFrame
        predictions = model.run(train, test)
        temp_res.update(run_metrics(predictions, test, metrics, per_sens_metrics))
        results = results.append(temp_res, ignore_index=True)
        print(f"seed_{seed}_{model.name}_completed")

    outdir = Path("..") / "results"  # OS-independent way of saying '../results'
    outdir.mkdir(exist_ok=True)
    path_to_file = outdir / f"seed_{seed}_beutel_reduced.csv"
    exists = os.path.isfile(path_to_file)
    if exists:
        loaded_results = pd.read_csv(path_to_file)
        results = pd.concat([loaded_results, results])
    results.to_csv(path_to_file, index=False)


def main():
    SEED = int(sys.argv[1])

    data_loc = Path(".") / "data" / "styling_beutel" / f"seed_{SEED}"

    train_beutel_dataset: Dataset = create_data_obj(
        data_loc / f"seed_{SEED}_stylingtraintilde.csv",
        s_columns=["sensitive"],
        y_columns=["label"],
    )
    train_beutel_data: DataTuple = load_data(train_tilde_dataset)

    test_beutel_dataset: Dataset = create_data_obj(
        data_loc / f"seed_{SEED}_stylingtesttilde.csv", s_columns=["sensitive"], y_columns=["label"]
    )
    test_beutel_data: DataTuple = load_data(test_tilde_dataset)

    run_experiments(train_beutel_data, test_beutel_data, "tilde", SEED)

    print(f"finished seed {SEED}")


if __name__ == "__main__":
    main()
