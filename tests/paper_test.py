"""Run experiments as tests."""

import pandas as pd
from sklearn.preprocessing import StandardScaler

from ethicml import (
    LRCV,
    TNR,
    TPR,
    Accuracy,
    BalancedTestSplit,
    DataTuple,
    Kamiran,
    ProbPos,
    diff_per_sensitive_attribute,
    metric_per_sensitive_attribute,
    save_2d_plot,
    save_label_plot,
    save_multijointplot,
    train_test_split,
)
from ethicml.algorithms.inprocess.facct import Facct
from ethicml.data.tabular_data.third_way import third_way


def test_plots():
    """Generate some plots to try to understand the data a bit more."""
    data = third_way().load()
    save_label_plot(data, "./data_original.png")
    balanced, unbalanced, _ = BalancedTestSplit(balance_type="P(s,y)=0.25")(data)
    save_label_plot(balanced, "./data_biased.png")
    save_label_plot(unbalanced, "./data_unbiased.png")

    save_2d_plot(data, "./data_2dplot.png")
    save_multijointplot(data, "./multiplot.png")


def test_lr():
    """Test the LRCV model."""
    data = third_way().load()
    train, test = train_test_split(data)

    preds = LRCV().run(train, test)
    accuracy = Accuracy().score(preds, test)
    print(f"{accuracy}")
    print(ProbPos().score(preds, test))

    print(f"Acc: {metric_per_sensitive_attribute(preds, test, Accuracy())}")
    print(diff_per_sensitive_attribute(metric_per_sensitive_attribute(preds, test, Accuracy())))
    print(f"DP: {metric_per_sensitive_attribute(preds, test, ProbPos())}")
    print(diff_per_sensitive_attribute(metric_per_sensitive_attribute(preds, test, ProbPos())))
    print(f"TPR: {metric_per_sensitive_attribute(preds, test, TPR())}")
    print(diff_per_sensitive_attribute(metric_per_sensitive_attribute(preds, test, TPR())))
    print(f"TNR: {metric_per_sensitive_attribute(preds, test, TNR())}")
    print(diff_per_sensitive_attribute(metric_per_sensitive_attribute(preds, test, TNR())))


def test_lr_wys_wae():
    """Training set is random sample, test set balanced. Model is LRCV."""
    data = third_way().load()
    train, test, _ = BalancedTestSplit(train_percentage=0.7)(data)

    preds = LRCV().run(train, test)
    accuracy = Accuracy().score(preds, test)
    print(f"{accuracy}")
    print(ProbPos().score(preds, test))
    print(f"DP: {metric_per_sensitive_attribute(preds, test, ProbPos())}")
    print(diff_per_sensitive_attribute(metric_per_sensitive_attribute(preds, test, ProbPos())))
    print(f"TPR: {metric_per_sensitive_attribute(preds, test, TPR())}")
    print(diff_per_sensitive_attribute(metric_per_sensitive_attribute(preds, test, TPR())))


def test_kc_wys_wae():
    """Training set is random sample, test set balanced. Model is K&C."""
    data = third_way().load()
    data = data.replace(x=pd.concat([data.x, data.s.rename({"sens": "s"}, axis=1)], axis=1))
    train, test, _ = BalancedTestSplit(train_percentage=0.7)(data)

    preds = Kamiran().run(train, test)
    accuracy = Accuracy().score(preds, test)
    print(f"{accuracy}")
    print(ProbPos().score(preds, test))
    print(f"DP: {metric_per_sensitive_attribute(preds, test, ProbPos())}")
    print(f"TPR: {metric_per_sensitive_attribute(preds, test, TPR())}")


def test_facct():
    """Test Facct Submission."""
    data = third_way().load()
    train, test = train_test_split(data)

    scaler = StandardScaler()
    scaler = scaler.fit(train.x)
    train = train.replace(x=pd.DataFrame(scaler.transform(train.x), columns=train.x.columns))
    test = test.replace(x=pd.DataFrame(scaler.transform(test.x), columns=test.x.columns))

    for model in [
        Facct(
            enc_epochs=50,
            clf_epochs=50,
            batch_size=64,
            enc_ld=10,
            pred_ld=10,
            wandb=0,
            warmup_steps=10,
        ),
        LRCV(),
    ]:
        print(f"=== {model.name} ===")
        results = model.run(train, test)

        print(results.hard.value_counts())
        print(Accuracy().score(results, test))
        print(f"Acc: {metric_per_sensitive_attribute(results, test, Accuracy())}")
        print(f"DP: {metric_per_sensitive_attribute(results, test, ProbPos())}")
        print(f"TPR: {metric_per_sensitive_attribute(results, test, TPR())}")

        if model.name == "Facct":
            data = DataTuple(x=test.x, s=test.s, y=pd.DataFrame(results.hard, columns=["y"]))
            save_label_plot(data, "./label_plot_ours.png")
            data = DataTuple(x=test.x, s=test.s, y=test.y)
            save_label_plot(data, "./label_plot_truth.png")


def test_facct_wys_wae():
    """Test agarwal."""
    data = third_way().load()
    train, test, _ = BalancedTestSplit(balance_type="P(s,y)=0.25", train_percentage=0.5)(data)

    scaler = StandardScaler()
    scaler = scaler.fit(train.x)
    train = train.replace(x=pd.DataFrame(scaler.transform(train.x), columns=train.x.columns))
    test = test.replace(x=pd.DataFrame(scaler.transform(test.x), columns=test.x.columns))

    for model in [Facct(epochs=50, batch_size=64, enc_ld=10, pred_ld=2), LRCV()]:
        print(f"=== {model.name} ===")
        results = model.run(train, test)

        print(results.hard.value_counts())
        print(Accuracy().score(results, test))
        print(f"DP: {metric_per_sensitive_attribute(results, test, ProbPos())}")
        print(f"TPR: {metric_per_sensitive_attribute(results, test, TPR())}")

        if model.name == "Facct":
            data = DataTuple(x=test.x, s=test.s, y=pd.DataFrame(results.hard, columns=["y"]))
            save_label_plot(data, "./data_label_inverstigatin.png")
            data = DataTuple(x=test.x, s=test.s, y=test.y)
            save_label_plot(data, "__data_label_inverstigatin_og.png")


if __name__ == '__main__':
    test_facct()
