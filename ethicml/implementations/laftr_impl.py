"""Model for generating sort of counterfactuals."""
import random
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader

from ethicml import LRCV, Accuracy, DataTuple, Majority, Prediction, TestTuple, Union
from ethicml.algorithms.inprocess.blind import Blind

from .laftr_modules.autoencoder import LaftrAE
from .laftr_modules.common import get_device
from .laftr_modules.data_lookup import data_lookup
from .laftr_modules.dataset import DataTupleDataset
from .utils import InAlgoArgs


class Datasets(str, Enum):
    """Add options for cli."""

    adult = "adult"
    third = "third"
    third_cf = "third_cf"
    sklearn = "sklearn"
    health = "health"


class LaftrArgs(InAlgoArgs):
    """Args for Facct."""

    # General args
    batch_size: int
    dataset: str
    device: int
    lr: float
    seed: int
    warmup_steps: int
    weight_decay: float

    # Autoencoder (Enc) args
    enc_additional_adv_steps: int
    enc_adv_weight: float
    enc_blocks: int
    enc_epochs: int
    enc_hidden_multiplier: int
    enc_ld: int
    enc_pred_weight: float
    enc_reg_weight: float


def load_ethicml_data(datatuple: Union[DataTuple, TestTuple], dataset: str) -> DataTupleDataset:
    """Load an EthicML Dataset."""
    dataset_obj = data_lookup(dataset)
    disc_feature_groups = dataset_obj.disc_feature_groups
    if disc_feature_groups is None:
        raise ValueError("Can only run on datasets with feature groups.")
    cont_features = dataset_obj.cont_features
    return DataTupleDataset(
        datatuple, disc_feature_groups=disc_feature_groups, cont_features=cont_features
    )


def train_and_predict(train: DataTuple, test: TestTuple, args: LaftrArgs) -> pd.Series:
    """Train and encoder for X, then Y, then group and predict."""
    device = get_device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False

    # Set up the data
    train_data = load_ethicml_data(train, dataset=args.dataset)
    train_loader = DataLoader(train_data, batch_size=args.batch_size)
    test_data = load_ethicml_data(test, dataset=args.dataset)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    encoder = LaftrAE(
        in_size=train_data.xdim,
        latent_dim=args.enc_ld,
        blocks=args.enc_blocks,
        hidden_multiplier=args.enc_hidden_multiplier,
        feature_groups=train_data.disc_feature_group_slices,
    )
    encoder.device = device
    enc_optimizer = Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    encoder.optim = enc_optimizer
    enc_scheduler = lr_scheduler.ExponentialLR(enc_optimizer, gamma=0.99)
    encoder = encoder.fit(
        data_loader=train_loader,
        epochs=args.enc_epochs,
        device=device,
        warmup_steps=args.warmup_steps,
        reg_weight=args.enc_reg_weight,
        adv_weight=args.enc_adv_weight,
        scheduler=enc_scheduler,
        additional_adv_steps=args.enc_additional_adv_steps,
        pred_weight=args.enc_pred_weight,
    )
    encoder.eval()

    # Generate both S samples
    outcomes = pd.DataFrame()
    for (x, s, _) in test_loader:
        x = x.to(device)
        s = s.to(device)
        outcome = pd.DataFrame()
        outcome["preds"] = encoder.predict(x, s)
        outcomes = outcomes.append(outcome)

    outcomes = outcomes.reset_index(drop=True)

    random_state = np.random.RandomState(seed=args.seed)
    folder = KFold(n_splits=5, shuffle=True, random_state=random_state)

    latent_train = encoder.get_latent(train_loader)
    latent_test = encoder.get_latent(test_loader)
    clf = LogisticRegressionCV(
        cv=folder, n_jobs=-1, random_state=random_state, solver="liblinear", multi_class="auto"
    )
    clf.fit(latent_train, train.s.to_numpy().ravel())
    s_preds = Prediction(hard=pd.Series(clf.predict(latent_test)), info=dict(C=clf.C_[0]))
    print(
        f"Acc at predicting S from Enc Z: "
        f"{Accuracy().score(prediction=s_preds, actual=DataTuple(x=test.x, s= test.s, y=test.s))}"
    )

    latent_train = encoder.get_recon(train_loader)
    latent_test = encoder.get_recon(test_loader)
    clf = LogisticRegressionCV(
        cv=folder, n_jobs=-1, random_state=random_state, solver="liblinear", multi_class="auto"
    )
    clf.fit(latent_train, train.s.to_numpy().ravel())
    s_preds = Prediction(hard=pd.Series(clf.predict(latent_test)), info=dict(C=clf.C_[0]))
    print(
        f"Acc at predicting S from Recon X: "
        f"{Accuracy().score(prediction=s_preds, actual=DataTuple(x=test.x, s= test.s, y=test.s))}"
    )

    for model in [LRCV, Blind, Majority]:
        s_preds = model().run(train.replace(y=train.s), test)
        print(
            f"{model().name} Acc at predicting S from X: "
            f"{Accuracy().score(prediction=s_preds, actual=DataTuple(x=test.x, s= test.s, y=test.s))}"
        )

    return outcomes["preds"]


def main() -> None:
    """This function runs the Facct model as a standalone program."""
    args: LaftrArgs = LaftrArgs().parse_args()
    train, test = DataTuple.from_npz(Path(args.train)), TestTuple.from_npz(Path(args.test))
    Prediction(hard=train_and_predict(train, test, args)).to_npz(Path(args.predictions))


if __name__ == "__main__":
    main()
