"""Model for generating sort of counterfactuals."""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

import wandb
from ethicml import (
    TNR,
    TPR,
    Accuracy,
    DataTuple,
    Prediction,
    ProbPos,
    metric_per_sensitive_attribute,
)
from ethicml.implementations.facct_modules.common import get_device
from ethicml.implementations.facct_modules.facct_classifier import FacctClassifier
from ethicml.implementations.facct_modules.facct_encoder import FacctAE
from ethicml.implementations.pytorch_common import CustomDataset, TestDataset
from ethicml.implementations.utils import InAlgoArgs


class FacctArgs(InAlgoArgs):
    """Args for Facct."""

    batch_size: int
    enc_epochs: int
    clf_epochs: int
    enc_ld: int
    pred_ld: int
    wandb: int
    warmup_steps: int
    device: int


def train_encoder(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    log_wandb: bool,
    device: torch.device,
    warmup: bool = False,
) -> nn.Module:
    """Train the AE."""
    model.train()
    train_loss = 0.0
    for batch_idx, (data_x, data_s, data_y) in enumerate(train_loader):
        optimizer.zero_grad()

        data_x = data_x.to(device)
        data_s = data_s.to(device)

        z, s_pred, x_pred = model(data_x, data_s)

        feat_sens_loss = F.binary_cross_entropy_with_logits(s_pred, data_s, reduction="mean")
        recon_loss = F.mse_loss(x_pred, data_x, reduction="mean")

        adv_weight = 0.0 if warmup else 0.5
        loss = 1.0 * recon_loss + adv_weight * feat_sens_loss + 1e-3 * torch.norm(z)

        if log_wandb:
            wandb.log(
                {
                    "enc_recon_loss": recon_loss,
                    "enc_adv_loss": feat_sens_loss,
                    "enc_latent_norm": torch.norm(z),
                }
            )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if not warmup:
            for _ in range(0):
                optimizer.zero_grad()
                z, s_pred, x_pred = model(data_x, data_s)
                loss = F.binary_cross_entropy_with_logits(s_pred, data_s, reduction="mean") / 3
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        train_loss += loss.item()

    model.eval()
    if log_wandb:
        wandb.log({"enc_loss": train_loss})
    return model


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    log_wandb: bool,
    device: torch.device,
    warmup: bool = False,
) -> nn.Module:
    """Train the classifier."""
    model.train()
    train_loss = 0.0

    for batch_idx, (data_x, data_s, data_y) in enumerate(train_loader):
        optimizer.zero_grad()

        data_x = data_x.to(device)
        data_s = data_s.to(device)
        data_y = data_y.to(device)

        z, s_pred, y_pred = model(data_x, data_s)

        feat_sens_loss = F.binary_cross_entropy_with_logits(s_pred, data_s, reduction="mean")
        pred_y_loss = F.binary_cross_entropy_with_logits(y_pred, data_y, reduction="mean")

        adv_weight = 0.0 if warmup else 0.5
        loss = 1.0 * pred_y_loss + adv_weight * feat_sens_loss + 0.01 * torch.norm(z)

        if log_wandb:
            wandb.log(
                {
                    "clf_pred_loss": pred_y_loss,
                    "clf_adv_loss": feat_sens_loss,
                    "clf_latent_norm": torch.norm(z),
                }
            )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

        if not warmup:
            for _ in range(0):
                optimizer.zero_grad()
                z, s_pred, y_pred = model(data_x, data_s)
                loss = F.binary_cross_entropy_with_logits(s_pred, data_s, reduction="mean")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    model.eval()
    if log_wandb:
        wandb.log({"clf_loss": train_loss / len(train_loader.dataset)})
    return model


def train_and_predict(train: DataTuple, test: DataTuple, args: FacctArgs) -> pd.Series:
    """Train and encoder for X, then Y, then group and predict."""
    random.seed(888)
    np.random.seed(888)
    torch.random.manual_seed(888)

    device = get_device(args.device)

    # Set up the data
    train_data = CustomDataset(train)
    train_loader = DataLoader(train_data, batch_size=args.batch_size)
    test_data = TestDataset(test)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    # Build Encoder Network
    encoder = FacctAE(in_size=train_data.xdim, latent_dim=args.enc_ld).to(device)
    enc_optimizer = Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-6)
    enc_scheduler = lr_scheduler.ExponentialLR(enc_optimizer, gamma=0.98)

    # Train Encoder Network
    for epoch in range(int(args.enc_epochs)):
        encoder = train_encoder(
            model=encoder,
            train_loader=train_loader,
            optimizer=enc_optimizer,
            log_wandb=args.wandb > 0,
            warmup=epoch < args.warmup_steps,
            device=device,
        )
        enc_scheduler.step()

    # Build Classifier Network
    classifier = FacctClassifier(in_size=train_data.xdim, latent_dim=args.pred_ld).to(device)
    clf_optimizer = Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-6)
    clf_scheduler = lr_scheduler.ExponentialLR(clf_optimizer, gamma=0.98)

    # Train Classifier Network
    train.y.mean().values[0]
    for epoch in range(int(args.clf_epochs)):
        classifier = train_classifier(
            model=classifier,
            train_loader=train_loader,
            optimizer=clf_optimizer,
            log_wandb=args.wandb > 0,
            warmup=epoch < args.warmup_steps,
            device=device,
        )
        clf_scheduler.step()

    # Generate both S samples
    outcomes = pd.DataFrame()
    for (x, s) in test_loader:
        outcome = pd.DataFrame()
        for s1_label in [torch.zeros_like(s), torch.ones_like(s)]:
            _, _, x_pred = encoder(x, s1_label)
            for s2_label in [torch.zeros_like(s), torch.ones_like(s)]:
                _, _, yh = classifier(x_pred, s2_label)
                outcome[f"s1:{int(s1_label[0].item())}_s2:{int(s2_label[0].item())}"] = (
                    yh.sigmoid().round().cpu().detach().numpy()
                ).flatten()
        outcome['true_s'] = f"s1:{int(s[0].item())}_s2:{int(s[0].item())}"
        outcome['actual'] = outcome[f"s1:{int(s[0].item())}_s2:{int(s[0].item())}"]

        outcomes = outcomes.append(outcome)

    outcomes = outcomes.reset_index(drop=True)
    outcomes["preds"] = outcomes["actual"]  # outcomes.sum(axis=1)

    consensus = Prediction(
        hard=outcomes[outcomes["s1:0_s2:0"] == outcomes["s1:1_s2:1"]]["s1:1_s2:1"]
    )

    truth = DataTuple(
        x=test.x[outcomes["s1:0_s2:0"] == outcomes["s1:1_s2:1"]],
        s=test.s[outcomes["s1:0_s2:0"] == outcomes["s1:1_s2:1"]],
        y=test.y[outcomes["s1:0_s2:0"] == outcomes["s1:1_s2:1"]],
    )

    print(Accuracy().score(consensus, truth))
    print(f"Acc: {metric_per_sensitive_attribute(consensus, truth, Accuracy())}")
    print(f"DP: {metric_per_sensitive_attribute(consensus, truth, ProbPos())}")
    print(f"TPR: {metric_per_sensitive_attribute(consensus, truth, TPR())}")
    print(f"TNR: {metric_per_sensitive_attribute(consensus, truth, TNR())}")

    random_state = np.random.RandomState(seed=888)
    folder = KFold(n_splits=5, shuffle=True, random_state=random_state)
    clf = LogisticRegressionCV(
        cv=folder, n_jobs=-1, random_state=random_state, solver="liblinear", multi_class="auto"
    )
    clf.fit(train.x, train.y.to_numpy().ravel())

    baseline = Prediction(
        hard=pd.Series(clf.predict(test.x))[outcomes["s1:0_s2:0"] == outcomes["s1:1_s2:1"]]
    )

    print(Accuracy().score(baseline, truth))
    print(f"Acc: {metric_per_sensitive_attribute(baseline, truth, Accuracy())}")
    print(f"DP: {metric_per_sensitive_attribute(baseline, truth, ProbPos())}")
    print(f"TPR: {metric_per_sensitive_attribute(baseline, truth, TPR())}")
    print(f"TNR: {metric_per_sensitive_attribute(baseline, truth, TNR())}")

    conditions = [
        (outcomes["s1:0_s2:0"] == outcomes["s1:1_s2:1"]),
        (outcomes["s1:0_s2:0"] <= 0.5)
        & (outcomes["s1:0_s2:1"] <= 0.5)
        & (outcomes["s1:1_s2:0"] >= 0.5)
        & (outcomes["s1:1_s2:1"] >= 0.5),
        (outcomes["s1:0_s2:0"] >= 0.5)
        & (outcomes["s1:0_s2:1"] >= 0.5)
        & (outcomes["s1:1_s2:0"] <= 0.5)
        & (outcomes["s1:1_s2:1"] <= 0.5),
    ]
    values = ["consensus", "third", "unexpected"]
    outcomes["decision"] = np.select(conditions, values, "default value")
    print(outcomes["decision"].value_counts())

    return outcomes["preds"]


def main():
    """This function runs the Facct model as a standalone program."""
    args: FacctArgs = FacctArgs().parse_args()
    if args.wandb > 0:
        wandb.init(
            project="facct",
            # config={k: getattr(args, k) for k in list(vars(FacctArgs)['__annotations__'])},
        )
    train, test = DataTuple.from_npz(Path(args.train)), DataTuple.from_npz(Path(args.test))
    Prediction(hard=train_and_predict(train, test, args)).to_npz(Path(args.predictions))


if __name__ == "__main__":
    main()
