"""
Wrapper for calling the fair GP model
"""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from ethicml.algorithms.inprocess.installed_model import InstalledModel

PRED_FNAME = 'predictions.npz'
MAX_EPOCHS = 1000
MAX_BATCH_SIZE = 10100  # can go up to 10000
MAX_NUM_INDUCING = 5000  # 2500 seems to be more than enough
SEED = 1234


class GPyT(InstalledModel):
    """
    Normal GP model
    """
    basename = "GPyT"

    def __init__(self, s_as_input=True, gpu=0, epochs=70):
        super().__init__(name="gpyt",
                         url="https://github.com/predictive-analytics-lab/fair-gpytorch.git",
                         module="fair-gpytorch",
                         file_name="run.py")
        self.s_as_input = s_as_input
        self.gpu = gpu
        self.epochs = epochs

    def run(self, train, test, _=False):
        (ytrain, ytest), label_converter = _fix_labels([train.y.to_numpy(), test.y.to_numpy()])
        raw_data = dict(xtrain=train.x.to_numpy(), strain=train.s.to_numpy(), ytrain=ytrain,
                        xtest=test.x.to_numpy(), stest=test.s.to_numpy(), ytest=ytest)
        parameters = self._additional_parameters(raw_data)

        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            data_path = tmp_path / "data.npz"
            np.savez(data_path, **raw_data)
            model_name = "local"
            flags = _flags(parameters, str(data_path), tmpdir, self.s_as_input, model_name,
                           raw_data['ytrain'].shape[0], self.gpu, self.epochs)
            self._run_gpyt(flags)

            # Read the results from the numpy file 'predictions.npz'
            with (tmp_path / model_name / PRED_FNAME).open('rb') as file_obj:
                output = np.load(file_obj)
                pred_mean = output['pred_mean']

        predictions = label_converter((pred_mean > 0.5).astype(raw_data['ytest'].dtype)[:, 0])
        return pd.DataFrame(predictions, columns=['preds'])

    def _run_gpyt(self, flags):
        """Generate command to run GPyT"""
        cmd = [str(self._module_path() / self.file_name)]
        for key, value in flags.items():
            cmd += [f"--{key}", str(value)]
        self._call_script(cmd)

    @staticmethod
    def _additional_parameters(_):
        return dict(
            lik='BaselineLikelihood',
        )

    @property
    def name(self):
        return f"{self.basename}_in_{self.s_as_input}"


def _fix_labels(labels):
    """Make sure that labels are either 0 or 1

    Args"
        labels: the labels as a list of numpy arrays
        positive_class_val: the value that corresponds to a "positive" predictions

    Returns:
        the fixed labels and a function to convert the fixed labels back to the original format
    """
    label_values = list(np.unique(labels[0]))
    if label_values == [0, 1]:

        def _do_nothing(inp):
            return inp
        return labels, _do_nothing
    if label_values == [-1, 1]:

        def _converter(label):
            return (2 * label) - 1
        return [(y + 1) / 2 for y in labels], _converter
    raise ValueError("Labels have unknown structure")


def _flags(parameters, data_path, save_dir, s_as_input, model_name, num_train, gpu, epochs):
    batch_size = min(MAX_BATCH_SIZE, num_train)
    epochs = _num_epochs(num_train) if epochs is None else epochs
    return {**dict(
        inf='Variational',
        data='sensitive_from_numpy',
        dataset_path=data_path,
        cov='RBFKernel',
        mean='ZeroMean',
        optimizer="Adam",
        lr=0.05,
        # lr=0.1,
        model_name=model_name,
        batch_size=batch_size,
        # epochs=min(MAX_EPOCHS, _num_epochs(num_train)),
        epochs=min(epochs, MAX_EPOCHS),
        eval_epochs=5,
        summary_steps=100000,
        chkpt_epochs=100000,
        save_dir=save_dir,  # "/home/ubuntu/out2/",
        plot='',
        logging_steps=1,
        gpus=str(gpu),
        preds_path=PRED_FNAME,  # save the predictions into `predictions.npz`
        num_samples=1000,
        optimize_inducing=True,
        length_scale=1.2,
        sf=1.0,
        iso=False,
        num_samples_pred=2000,
        s_as_input=s_as_input,
        # num_inducing=MAX_NUM_INDUCING,
        num_inducing=_num_inducing(num_train),
        manual_seed=SEED,
        metrics=("binary_accuracy,pred_rate_y1_s0,pred_rate_y1_s1,base_rate_y1_s0,base_rate_y1_s1,"
                 "pred_odds_yhaty1_s0,pred_odds_yhaty1_s1,pred_odds_yhaty0_s0,pred_odds_yhaty0_s1")
    ), **parameters}


def _num_inducing(num_train):
    """Adaptive number of inducing inputs

    num_train == 4,000 => num_inducing == 1121
    num_train == 20,000 => num_inducing == 2507
    """
    return int(2500 / 141 * np.sqrt(num_train))


def _num_epochs(num_train):
    """Adaptive number of epochs

    num_train == 4,000 => num_epochs == 125.7
    num_train == 20,000 => num_epochs == 84
    """
    return int(1000 / np.power(num_train, 1 / 4))
