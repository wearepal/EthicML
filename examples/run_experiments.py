import os
from pathlib import Path
import pandas as pd

from ethicml.algorithms.inprocess import SVM, Kamishima, Agarwal
from ethicml.algorithms.utils import DataTuple
from ethicml.data.dataset import Dataset
from ethicml.data.load import create_data_obj, load_data
from ethicml.evaluators.cross_validator import CrossValidator
from ethicml.evaluators.evaluate_models import run_metrics
from ethicml.metrics import Accuracy, CV, ProbPos, TPR

SEED = 0

metrics = [Accuracy(), CV()]
per_sens_metrics = [Accuracy(), TPR(), ProbPos()]


def run_experiments(train, test, exp_name):
    svm_hyperparams = {'C': [10 ** x for x in range(-5, 5)], 'kernel': ['rbf', 'linear']}

    svm_cv = CrossValidator(SVM, svm_hyperparams, folds=3)
    svm_cv.run(train)
    best_svm = svm_cv.best(Accuracy())

    best_svm_hyperparams = svm_cv.best_hyper_params(Accuracy())


    lr_hyperparams = {'C': [10 ** x for x in range(-5, 5)]}

    lr_cv = CrossValidator(SVM, lr_hyperparams, folds=3)
    lr_cv.run(train)
    best_lr = lr_cv.best(Accuracy())

    best_lr_hyperparams = lr_cv.best_hyper_params(Accuracy())

    models = [best_svm, best_lr, Kamishima(), Agarwal(fairness="EqOd", classifier="LR",
                                                      hyperparams=best_lr_hyperparams),
              Agarwal(fairness="EqOd", classifier="SVM", hyperparams=best_svm_hyperparams),
              ]

    columns = ['dataset', 'transform', 'model', 'repeat']
    columns += [metric.name for metric in metrics]
    results = pd.DataFrame(columns=columns)

    for model in models:

        temp_res = {'dataset': "Adult",
                    'transform': exp_name,
                    'model': model.name,
                    'repeat': f"{SEED}"}

        predictions: pd.DataFrame
        predictions = model.run(train, test)

        temp_res.update(run_metrics(predictions, test, metrics, per_sens_metrics))

        results = results.append(temp_res, ignore_index=True)

    outdir = Path('..') / 'results'  # OS-independent way of saying '../results'
    outdir.mkdir(exist_ok=True)
    path_to_file = outdir / f"seed_{SEED}_{exp_name}.csv"
    exists = os.path.isfile(path_to_file)
    if exists:
        loaded_results = pd.read_csv(path_to_file)
        results = pd.concat([loaded_results, results])
    results.to_csv(path_to_file, index=False)


def main():
    data_loc = Path("~") / "Documents" / "styling_w_2_hsic" / "weight_100_decoder_1e-4" / f"seed_{SEED}"
    train_dataset: Dataset = create_data_obj(str(data_loc / f'seed_{SEED}_stylingtrain_50000.csv'),
                                             s_columns=["sensitive"],
                                             y_columns=["label"])
    train_data: DataTuple = load_data(train_dataset)

    train_tilde_dataset: Dataset = create_data_obj(str(data_loc / f'seed_{SEED}_stylingtraintilde_50000.csv'),
                                                   s_columns=["sensitive"],
                                                   y_columns=["label"])
    train_tilde_data: DataTuple = load_data(train_tilde_dataset)

    test_dataset: Dataset = create_data_obj(str(data_loc / f'seed_{SEED}_stylingtest_50000.csv'),
                                            s_columns=["sensitive"],
                                            y_columns=["label"])
    test_data: DataTuple = load_data(test_dataset)

    test_tilde_dataset: Dataset = create_data_obj(str(data_loc / f'seed_{SEED}_stylingtesttilde_50000.csv'),
                                                  s_columns=["sensitive"],
                                                  y_columns=["label"])
    test_tilde_data: DataTuple = load_data(test_tilde_dataset)

    train = train_data
    test = test_data
    run_experiments(train, test, "no_transform")

    train = train_tilde_data
    test = test_tilde_data
    run_experiments(train, test, "tilde")

    print(f'finished seed {SEED}')


if __name__ == "__main__":
    main()
