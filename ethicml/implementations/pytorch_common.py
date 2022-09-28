"""Functions that are common to PyTorch models."""
from __future__ import annotations
import random
from typing import Literal

import numpy as np
import pandas as pd

from ethicml.utility import DataTuple, TestTuple

try:
    import torch
    from torch import Tensor, nn
    from torch.utils.data import Dataset, TensorDataset
except ImportError as e:
    raise RuntimeError(
        "In order to use PyTorch, please install it following the instructions as https://pytorch.org/ . "
    ) from e


def _get_info(data: TestTuple) -> tuple[np.ndarray, np.ndarray, int, int, pd.Index, str]:
    features = data.x.to_numpy(dtype=np.float32)
    sens_labels = data.s.to_numpy(dtype=np.float32)
    num = data.s.shape[0]
    xdim = data.x.shape[1]
    x_names = data.x.columns
    s_name = str(data.s.name)
    return features, sens_labels, num, xdim, x_names, s_name


class TestDataset(Dataset):
    """Shared Dataset for pytorch models without labels."""

    def __init__(self, data: TestTuple):
        super().__init__()
        self.x, self.s, self.num, self.xdim, self.x_names, self.s_names = _get_info(data)
        self.sdim = 1

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Implement __getitem__ magic method."""
        return self.x[index, ...], self.s[index, ...]

    def __len__(self) -> int:
        """Implement __len__ magic method."""
        return self.num

    @property
    def names(self) -> tuple[pd.Index, str]:
        """Get tuple of x names and s names."""
        return self.x_names, self.s_names


class CustomDataset(Dataset):
    """Shared Dataset for pytorch models."""

    def __init__(self, data: DataTuple):
        super().__init__()
        test = data.remove_y()
        self.x, self.s, self.num, self.xdim, self.x_names, self.s_names = _get_info(test)
        self.sdim = 1
        self.y = data.y.to_numpy(dtype=np.float32)
        self.ydim = data.y.nunique()
        self.y_names = str(data.y.name)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Implement __getitem__ magic method."""
        return self.x[index, ...], self.s[index, ...], self.y[index, ...]

    def __len__(self) -> int:
        """Implement __len__ magic method."""
        return self.num

    @property
    def names(self) -> tuple[pd.Index, str, str]:
        """Get tuple of x names, s names and y names."""
        return self.x_names, self.s_names, self.y_names


def make_dataset_and_loader(
    data: DataTuple, *, batch_size: int, shuffle: bool, seed: int, drop_last: bool
) -> tuple[CustomDataset, torch.utils.data.DataLoader]:
    """Given a datatuple, create a dataset and a corresponding dataloader."""

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)

    dataset = CustomDataset(data)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return dataset, dataloader


def quadratic_time_mmd(x: Tensor, y: Tensor, sigma: float) -> Tensor:
    """Calculate MMD betweer 2 tensors of equal size.

    :param x: Sample 1.
    :param y: Sample 2.
    :param sigma: Scale of the RBF kernel.
    :returns: Tensor of MMD in each dim.
    """
    xx_gm = x @ x.t()
    xy_gm = x @ y.t()
    yy_gm = y @ y.t()
    x_sqnorms = torch.diagonal(xx_gm)
    y_sqnorms = torch.diagonal(yy_gm)

    def pad_first(x: Tensor) -> Tensor:
        return torch.unsqueeze(x, 0)

    def pad_second(x: Tensor) -> Tensor:
        return torch.unsqueeze(x, 1)

    gamma = 1 / (2 * sigma**2)
    # use the second binomial formula
    kernel_xx = torch.exp(-gamma * (-2 * xx_gm + pad_second(x_sqnorms) + pad_first(x_sqnorms)))
    kernel_xy = torch.exp(-gamma * (-2 * xy_gm + pad_second(x_sqnorms) + pad_first(y_sqnorms)))
    kernel_yy = torch.exp(-gamma * (-2 * yy_gm + pad_second(y_sqnorms) + pad_first(y_sqnorms)))

    xx_num = float(kernel_xx.shape[0])
    yy_num = float(kernel_yy.shape[0])

    mmd2 = (
        kernel_xx.sum() / (xx_num * xx_num)
        + kernel_yy.sum() / (yy_num * yy_num)
        - 2 * kernel_xy.sum() / (xx_num * yy_num)
    )
    return mmd2


def compute_projection_gradients(
    model: nn.Module, loss_p: Tensor, loss_a: Tensor, alpha: float
) -> None:
    """Compute the adversarial gradient projection term.

    :param model: Model whose parameters the gradients are to be computed w.r.t.
    :param loss_p: Prediction loss.
    :param loss_a: Adversarial loss.
    :param alpha: Pre-factor for adversarial loss.
    """
    grad_p = torch.autograd.grad(loss_p, model.parameters(), retain_graph=True)  # type: ignore[arg-type]
    grad_a = torch.autograd.grad(loss_a, model.parameters(), retain_graph=True)  # type: ignore[arg-type]

    def _proj(a: Tensor, b: Tensor) -> Tensor:
        return b * torch.sum(a * b) / torch.sum(b * b)

    grad_p = tuple(p - _proj(p, a) - alpha * a for p, a in zip(grad_p, grad_a))

    for param, grad in zip(model.parameters(), grad_p):
        param.grad = grad


class PandasDataSet(TensorDataset):
    """Pandas Dataset."""

    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super().__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame('dummy')
        return torch.from_numpy(df.values).float()


class LinearModel(torch.nn.Module):
    """Define linear model."""

    def __init__(self, in_shape: int = 1, out_shape: int = 2):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.base_model = nn.Sequential(
            nn.Linear(self.in_shape, self.out_shape, bias=True),
        )

    def build_model(self) -> None:
        """Build Model."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return torch.squeeze(self.base_model(x))


class DeepModel(torch.nn.Module):
    """Define deep neural net model for classification."""

    def __init__(self, in_shape: int = 1, out_shape: int = 1):
        super().__init__()
        self.in_shape = in_shape
        self.dim_h = 64
        self.dropout = 0.5
        self.out_shape = out_shape
        self.build_model()

    def build_model(self) -> None:
        """Build Model."""
        self.base_model = nn.Sequential(
            nn.Linear(self.in_shape, self.dim_h, bias=True),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dim_h, self.out_shape, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return torch.squeeze(self.base_model(x))


class DeepRegModel(torch.nn.Module):
    """Define deep model for regression."""

    def __init__(self, in_shape: int = 1, out_shape: int = 1):
        super().__init__()
        self.in_shape = in_shape
        self.dim_h = 64  # in_shape*10
        self.out_shape = out_shape
        self.build_model()

    def build_model(self) -> None:
        """Build model."""
        self.base_model = nn.Sequential(
            nn.Linear(self.in_shape, self.dim_h, bias=True),
            nn.ReLU(),
            nn.Linear(self.dim_h, self.out_shape, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return torch.squeeze(self.base_model(x))


class DeepProbaModel(torch.nn.Module):
    """Define deep regression model, used by the fair dummies test."""

    def __init__(self, in_shape: int = 1):
        super().__init__()
        self.in_shape = in_shape
        self.dim_h = 64  # in_shape*10
        self.dropout = 0.5
        self.out_shape = 1
        self.build_model()

    def build_model(self) -> None:
        """Build Model."""
        self.base_model = nn.Sequential(
            nn.Linear(self.in_shape, self.dim_h, bias=True),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.dim_h, self.out_shape, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return torch.squeeze(self.base_model(x))


class GeneralLearner:
    """General Learner."""

    def __init__(
        self,
        lr: float,
        epochs: int,
        cost_func: nn.Module,
        in_shape: int,
        batch_size: int,
        model_type: Literal["deep_proba", "deep_regression"],
        out_shape: int = 1,
    ):

        # input dim
        self.in_shape = in_shape

        # output dim
        self.out_shape = out_shape

        # learning rate
        self.lr = lr

        # number of epochs
        self.epochs = epochs

        # cost to minimize
        self.cost_func = cost_func

        self.rng = np.random.default_rng(0)

        # define a predictive model
        self.model_type = model_type
        if self.model_type == "deep_proba":
            self.model: nn.Module = DeepProbaModel(in_shape=in_shape)
        elif self.model_type == "deep_regression":
            self.model = DeepModel(in_shape=in_shape, out_shape=self.out_shape)
        else:
            raise NotImplementedError

        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

        # minibatch size
        self.batch_size = batch_size

    def internal_epoch(self, dataloader: torch.utils.data.DataLoader) -> np.ndarray:
        """Fit a model by sweeping over all data points."""
        # fit pred func
        epoch_losses = []
        for x, s, y in dataloader:
            self.optimizer.zero_grad()
            # utility loss
            batch_yhat = self.model(x)
            loss = self.cost_func(batch_yhat, y.long())
            loss.backward()
            self.optimizer.step()
            epoch_losses.append(loss.cpu().detach().numpy())
        return np.mean(epoch_losses)

    def run_epochs(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Run epochs."""
        for _ in range(self.epochs):
            self.internal_epoch(dataloader)

    def fit(self, train: DataTuple, seed: int) -> None:
        """Fit a model on training data."""
        self.model.train()
        _, train_loader = make_dataset_and_loader(
            train, batch_size=self.batch_size, shuffle=True, seed=seed, drop_last=True
        )
        self.run_epochs(train_loader)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict output."""
        self.model.eval()
        xp = torch.from_numpy(x).float()
        yhat = self.model(xp)
        return yhat.detach().numpy()
