"""Implementation of Beutel's adversarially learned fair representations."""
# Disable pylint checking overwritten method signatures. Pytorch forward passes use **kwargs
# pylint: disable=arguments-differ

import random
from typing import Any, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.autograd import Function
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from ethicml.preprocessing.adjust_labels import LabelBinarizer, assert_binary_labels
from ethicml.preprocessing.train_test_split import train_test_split
from ethicml.utility import DataTuple, FairnessType, TestTuple

from .pytorch_common import CustomDataset, TestDataset
from .utils import PreAlgoArgs, load_data_from_flags, save_transformations

STRING_TO_ACTIVATION_MAP = {"Sigmoid()": nn.Sigmoid()}

STRING_TO_LOSS_MAP = {"BCELoss()": nn.BCELoss()}


class BeutelArgs(PreAlgoArgs):
    """Args for the Beutel Implementation."""

    fairness: FairnessType
    enc_size: List[int]
    adv_size: List[int]
    pred_size: List[int]
    enc_activation: str
    adv_activation: str
    batch_size: int
    y_loss: str
    s_loss: str
    epochs: int
    adv_weight: float
    validation_pcnt: float


def set_seed(seed: int):
    """Set the seeds for numpy torch etc."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]


def build_networks(flags: BeutelArgs, train_data: CustomDataset, enc_activation, adv_activation):
    """Build the networks we use.

    Pulled into a separate function to make the code a bit neater.
    """
    enc = Encoder(
        enc_size=flags.enc_size, init_size=int(train_data.xdim), activation=enc_activation
    )
    adv = Adversary(
        fairness=flags.fairness,
        adv_size=flags.adv_size,
        init_size=flags.enc_size[-1],
        s_size=int(train_data.sdim),
        activation=adv_activation,
        adv_weight=flags.adv_weight,
    )
    pred = Predictor(
        pred_size=flags.pred_size,
        init_size=flags.enc_size[-1],
        class_label_size=int(train_data.ydim),
        activation=adv_activation,
    )

    model = Model(enc, adv, pred)

    return enc, model


def make_dataset_and_loader(
    data: DataTuple, flags: BeutelArgs
) -> Tuple[CustomDataset, torch.utils.data.DataLoader]:
    """Given a datatuple, create a dataset and a corresponding dataloader.

    Args:
        data:
        flags:

    Returns:
        Tuple of a pytorch dataset and dataloader.
    """
    dataset = CustomDataset(data)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=flags.batch_size, shuffle=False
    )
    return dataset, dataloader


def train_and_transform(
    train: DataTuple, test: TestTuple, flags: BeutelArgs
) -> Tuple[DataTuple, TestTuple]:
    """Train the fair autoencoder on the training data and then transform both training and test."""
    set_seed(888)

    post_process = False
    if flags.y_loss == "BCELoss()":
        try:
            assert_binary_labels(train)
        except AssertionError:
            processor = LabelBinarizer()
            train = processor.adjust(train)
            post_process = True

    # By default we use 10% of the training data for validation
    train_, validation = train_test_split(train, train_percentage=1 - flags.validation_pcnt)

    train_data, train_loader = make_dataset_and_loader(train_, flags)
    _, validation_loader = make_dataset_and_loader(validation, flags)
    _, all_train_data_loader = make_dataset_and_loader(train, flags)

    test_data = TestDataset(test)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=flags.batch_size, shuffle=False
    )

    # convert flags to Python objects
    enc_activation = STRING_TO_ACTIVATION_MAP[flags.enc_activation]
    adv_activation = STRING_TO_ACTIVATION_MAP[flags.adv_activation]
    y_loss_fn = STRING_TO_LOSS_MAP[flags.y_loss]
    s_loss_fn = STRING_TO_LOSS_MAP[flags.s_loss]

    enc, model = build_networks(
        flags=flags,
        train_data=train_data,
        enc_activation=enc_activation,
        adv_activation=adv_activation,
    )

    optimizer = torch.optim.Adam(model.parameters())
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    best_val_loss = torch.ones(1) * np.inf
    best_enc = None

    for i in range(1, flags.epochs + 1):
        model.train()
        for embedding, sens_label, class_label in train_loader:
            _, s_pred, y_pred = model(embedding, class_label)

            loss = y_loss_fn(y_pred, class_label)

            if flags.fairness == "EqOp":
                mask = class_label.ge(0.5)
            elif flags.fairness == "EqOd":
                raise NotImplementedError("Not implemented Eq. Odds yet")
            elif flags.fairness == "DP":
                mask = torch.ones(s_pred.shape, dtype=torch.uint8)
            loss += s_loss_fn(
                s_pred, torch.masked_select(sens_label, mask).view(-1, int(train_data.sdim))
            )

            step(i, loss, optimizer, scheduler)

        if i % 5 == 0 or i == flags.epochs:
            model.eval()
            val_y_loss = torch.zeros(1)
            val_s_loss = torch.zeros(1)
            for embedding, sens_label, class_label in validation_loader:
                _, s_pred, y_pred = model(embedding, class_label)

                val_y_loss += y_loss_fn(y_pred, class_label)

                mask = get_mask(flags, s_pred, class_label)

                val_s_loss -= s_loss_fn(
                    s_pred, torch.masked_select(sens_label, mask).view(-1, int(train_data.sdim))
                )

            val_loss = (val_y_loss / len(validation_loader)) + (val_s_loss / len(validation_loader))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_enc = enc.state_dict()

    enc.load_state_dict(best_enc)

    transformed_train = encode_dataset(enc, all_train_data_loader, train)
    if post_process:
        transformed_train = processor.post(encode_dataset(enc, all_train_data_loader, train))
    return transformed_train, encode_testset(enc, test_loader, test)


def step(iteration: int, loss: Tensor, optimizer: Adam, scheduler: ExponentialLR) -> None:
    """Do one training step."""
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step(iteration)


def get_mask(flags: BeutelArgs, s_pred, class_label):
    """Get a mask to enforce different fairness types."""
    if flags.fairness == "EqOp":
        mask = class_label.ge(0.5)
    elif flags.fairness == "EqOd":
        raise NotImplementedError("Not implemented Eq. Odds yet")
    elif flags.fairness == "DP":
        mask = torch.ones(s_pred.shape, dtype=torch.uint8)

    return mask


def encode_dataset(
    enc: nn.Module, dataloader: torch.utils.data.DataLoader, datatuple: DataTuple
) -> DataTuple:
    """Encode a dataset."""
    data_to_return: List[Any] = []

    for embedding, _, _ in dataloader:
        data_to_return += enc(embedding).data.numpy().tolist()

    return DataTuple(
        x=pd.DataFrame(data_to_return),
        s=datatuple.s,
        y=datatuple.y,
        name=f"Beutel: {datatuple.name}",
    )


def encode_testset(
    enc: nn.Module, dataloader: torch.utils.data.DataLoader, testtuple: TestTuple
) -> TestTuple:
    """Encode a dataset.

    Args:
        enc:
        dataloader:

    Returns:
        Encoded TestTuple.
    """
    data_to_return: List[Any] = []

    for embedding, _ in dataloader:
        data_to_return += enc(embedding).data.numpy().tolist()

    return TestTuple(x=pd.DataFrame(data_to_return), s=testtuple.s, name=testtuple.name)


class GradReverse(Function):
    """Gradient reversal layer."""

    @staticmethod
    def forward(ctx, x: Tensor, lambda_: float) -> Tensor:  # type: ignore[override]
        """Forward pass."""
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass with Gradient reversed / inverted."""
        return grad_output.neg().mul(ctx.lambda_), None


def _grad_reverse(features: Tensor, lambda_: float) -> Tensor:
    return GradReverse.apply(features, lambda_)  # type: ignore[attr-defined]


class Encoder(nn.Module):
    """Encoder of the GAN."""

    def __init__(self, enc_size: Sequence[int], init_size: int, activation):
        super().__init__()
        self.encoder = nn.Sequential()
        if not enc_size:  # In the case that encoder size [] is specified
            self.encoder.add_module("single encoder layer", nn.Linear(init_size, init_size))
            self.encoder.add_module("single layer encoder activation", activation)
        else:
            self.encoder.add_module("encoder layer 0", nn.Linear(init_size, enc_size[0]))
            self.encoder.add_module("encoder activation 0", activation)
            for k in range(len(enc_size) - 1):
                self.encoder.add_module(
                    "encoder layer {}".format(k + 1), nn.Linear(enc_size[k], enc_size[k + 1])
                )
                self.encoder.add_module("encoder activation {}".format(k + 1), activation)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """Forward pass."""
        encoded = self.encoder(x)
        return encoded


class Adversary(nn.Module):
    """Adversary of the GAN."""

    def __init__(
        self,
        fairness: FairnessType,
        adv_size: Sequence[int],
        init_size: int,
        s_size: int,
        activation: nn.Module,
        adv_weight: float,
    ):
        super().__init__()
        self.fairness = fairness
        self.init_size = init_size
        self.adv_weight = adv_weight
        self.adversary = nn.Sequential()
        if not adv_size:  # In the case that encoder size [] is specified
            self.adversary.add_module("single adversary layer", nn.Linear(init_size, s_size))
            self.adversary.add_module("single layer adversary activation", activation)
        else:
            self.adversary.add_module("adversary layer 0", nn.Linear(init_size, adv_size[0]))
            self.adversary.add_module("adversary activation 0", activation)
            for k in range(len(adv_size) - 1):
                self.adversary.add_module(
                    "adversary layer {}".format(k + 1), nn.Linear(adv_size[k], adv_size[k + 1])
                )
                self.adversary.add_module("adversary activation {}".format(k + 1), activation)
            self.adversary.add_module("adversary last layer", nn.Linear(adv_size[-1], s_size))
            self.adversary.add_module("adversary last activation", activation)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:  # type: ignore[override]
        """Forward pass."""
        x = _grad_reverse(x, lambda_=self.adv_weight)

        if self.fairness == "EqOp":
            mask = y.ge(0.5)
            x = torch.masked_select(x, mask).view(-1, self.init_size)
            x = self.adversary(x)
        elif self.fairness == "EqOd":
            raise NotImplementedError("Not implemented equalized odds yet")
        elif self.fairness == "DP":
            x = self.adversary(x)
        return x


class Predictor(nn.Module):
    """Predictor of the GAN."""

    def __init__(
        self, pred_size: Sequence[int], init_size: int, class_label_size: int, activation: nn.Module
    ):
        super().__init__()
        self.predictor = nn.Sequential()
        if not pred_size:  # In the case that encoder size [] is specified
            self.predictor.add_module(
                "single adversary layer", nn.Linear(init_size, class_label_size)
            )
            self.predictor.add_module("single layer adversary activation", activation)
        else:
            self.predictor.add_module("adversary layer 0", nn.Linear(init_size, pred_size[0]))
            self.predictor.add_module("adversary activation 0", activation)
            for k in range(len(pred_size) - 1):
                self.predictor.add_module(
                    "adversary layer {}".format(k + 1), nn.Linear(pred_size[k], pred_size[k + 1])
                )
                self.predictor.add_module("adversary activation {}".format(k + 1), activation)
            self.predictor.add_module(
                "adversary last layer", nn.Linear(pred_size[-1], class_label_size)
            )
            self.predictor.add_module("adversary last activation", activation)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """Forward pass."""
        return self.predictor(x)


class Model(nn.Module):
    """Whole GAN model."""

    def __init__(self, enc: Encoder, adv: Adversary, pred: Predictor) -> None:
        super().__init__()
        self.enc = enc
        self.adv = adv
        self.pred = pred

    def forward(  # type: ignore[override]
        self, x: Tensor, y: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass."""
        encoded = self.enc(x)
        s_hat = self.adv(encoded, y)
        y_hat = self.pred(encoded)
        return encoded, s_hat, y_hat


def main():
    """Load data from feather files, pass it to `train_and_transform` and then save the result."""
    args = BeutelArgs().parse_args()
    train, test = load_data_from_flags(args)
    save_transformations(train_and_transform(train, test, args), args)


if __name__ == "__main__":
    main()
