from typing import Sequence, Tuple, List

import torch
from dataclasses import dataclass
from torch import nn, optim
from torch.autograd import Function
from torch.utils.data import DataLoader
import torch.distributions as td
import torch.nn.functional as F
import numpy as np
import pandas as pd

from ethicml.data import Dataset
from ethicml.implementations.beutel import set_seed
from ethicml.implementations.imagine_modules.adversary import FeatAdversary, \
    PredAdversary
from ethicml.implementations.imagine_modules.features import Features
from ethicml.implementations.imagine_modules.predictor import Predictor
from ethicml.implementations.pytorch_common import TestDataset, CustomDataset
from ethicml.implementations.utils import pre_algo_argparser, load_data_from_flags, \
    save_transformations
from ethicml.implementations.vfae import get_dataset_obj_by_name
from ethicml.preprocessing import LabelBinarizer
from ethicml.preprocessing.adjust_labels import assert_binary_labels
from ethicml.utility import DataTuple, TestTuple, Heaviside

PRED_LD = 1
FEAT_LD = 2


@dataclass(frozen=True)  # "frozen" makes it immutable
class ImagineSettings:
    """Settings for the Imagined Examples algorithm. This is basically a type-safe flag-object."""

    enc_size: Sequence[int]
    adv_size: Sequence[int]
    pred_size: Sequence[int]
    batch_size: int
    epochs: int
    adv_weight: float
    validation_pcnt: float
    dataset: str


def train_and_transform(train: DataTuple, test: TestTuple, flags: ImagineSettings) -> Tuple[DataTuple, TestTuple]:
    dataset = get_dataset_obj_by_name(flags.dataset)
    set_seed(888)

    post_process = False

    try:
        assert_binary_labels(train)
    except AssertionError:
        processor = LabelBinarizer()
        train = processor.adjust(train)
        post_process = True

    # Set up the data
    train_data = CustomDataset(train)
    train_loader = DataLoader(train_data, batch_size=flags.batch_size)

    test_data = TestDataset(test)
    test_loader = DataLoader(test_data, batch_size=flags.batch_size)

    # Build Network
    model = Imagine(data=train_data, dataset=dataset).to("cpu")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # Run Network
    for epoch in range(int(flags.epochs)):
        train_model(epoch, model, train_loader, optimizer, flags)
        scheduler.step(epoch)

    # Transform output
    actual_feats: List[List[float]] = []
    feats_train: List[List[float]] = []
    s_1_list: List[List[float]] = []
    s_2_list: List[List[float]] = []
    actual_labels: List[List[float]] = []
    direct_preds_train: List[List[float]] = []
    preds_train: List[List[float]] = []

    feats_test: List[List[float]] = []
    preds_test: List[List[float]] = []
    with torch.no_grad():
        for _x, _s, _y in train_loader:
            feat_enc, feat_dec, feat_s_pred, pred_enc, pred_dec, pred_s_pred, direct_prediction = model(_x, _s, _s)
            feats_train += feat_dec.data.tolist()
            direct_preds_train += direct_prediction.probs.data.tolist()
            preds_train += pred_dec.probs.data.tolist()
            s_1_list += _s.data.tolist()
            s_2_list += _s.data.tolist()
            actual_labels += _y.data.tolist()
            actual_feats += _x.data.tolist()

            feat_enc, feat_dec, feat_s_pred, pred_enc, pred_dec, pred_s_pred, direct_prediction = model(
                _x, (_s-1)**2, _s)
            feats_train += feat_dec.data.tolist()
            direct_preds_train += direct_prediction.probs.data.tolist()
            preds_train += pred_dec.probs.data.tolist()
            s_1_list += ((_s-1)**2).data.tolist()
            s_2_list += _s.data.tolist()
            actual_labels += _y.data.tolist()
            actual_feats += _x.data.tolist()

            feat_enc, feat_dec, feat_s_pred, pred_enc, pred_dec, pred_s_pred, direct_prediction = model(
                feat_dec, (_s-1)**2, (_s-1)**2)
            feats_train += feat_dec.data.tolist()
            direct_preds_train += direct_prediction.probs.data.tolist()
            preds_train += pred_dec.probs.data.tolist()
            s_1_list += ((_s-1)**2).data.tolist()
            s_2_list += ((_s-1)**2).data.tolist()
            actual_labels += _y.data.tolist()
            actual_feats += _x.data.tolist()

            feat_enc, feat_dec, feat_s_pred, pred_enc, pred_dec, pred_s_pred, direct_prediction = model(
                _x, _s, (_s - 1) ** 2)
            feats_train += feat_dec.data.tolist()
            direct_preds_train += direct_prediction.probs.data.tolist()
            preds_train += pred_dec.probs.data.tolist()
            s_1_list += _s.data.tolist()
            s_2_list += ((_s-1)**2).data.tolist()
            actual_labels += _y.data.tolist()
            actual_feats += _x.data.tolist()

        for _x, _s in test_loader:
            feat_enc, feat_dec, feat_s_pred, pred_enc, pred_dec, pred_s_pred, direct_prediction = model(_x, _s, _s)
            feats_test += feat_dec.data.tolist()
            preds_test += pred_dec.probs.data.tolist()

    feats = pd.DataFrame(feats_train, columns=train.x.columns)
    direct_labels = pd.DataFrame(direct_preds_train, columns=train.y.columns, dtype=np.int64)
    direct_labels = direct_labels.applymap(lambda x: 1 if x >= 0.5 else 0)
    s_1 = pd.DataFrame(s_1_list, columns=train.s.columns)
    s_2 = pd.DataFrame(s_2_list, columns=train.s.columns)
    actual_labels = pd.DataFrame(actual_labels, columns=train.y.columns)
    actual_feats = pd.DataFrame(actual_feats, columns=train.x.columns)

    labels = pd.DataFrame(preds_train, columns=train.y.columns, dtype=np.int64)
    labels = labels.applymap(lambda x: 1 if x >= 0.5 else 0)

    print(f"there are {train.y.count()} labels in the original training set and {labels.count()} in the augmented set")

    print(labels[(s_1['s'] == 0) & (labels['y'] == 0)].count(), actual_labels[(s_1['s'] == 0) & (actual_labels['y'] == 0)].count())
    print(labels[(s_1['s'] == 0) & (labels['y'] == 1)].count(), actual_labels[(s_1['s'] == 0) & (actual_labels['y'] == 1)].count())
    print(labels[(s_1['s'] == 1) & (labels['y'] == 0)].count(), actual_labels[(s_1['s'] == 1) & (actual_labels['y'] == 0)].count())
    print(labels[(s_1['s'] == 1) & (labels['y'] == 1)].count(), actual_labels[(s_1['s'] == 1) & (actual_labels['y'] == 1)].count())
    to_return = DataTuple(x=actual_feats, s=s_1, y=labels, name=f"Imagined: {train.name}")

    print(direct_labels[(s_1['s'] == 0) & (direct_labels['y'] == 0)].count(), actual_labels[(s_1['s'] == 0) & (actual_labels['y'] == 0)].count())
    print(direct_labels[(s_1['s'] == 0) & (direct_labels['y'] == 1)].count(), actual_labels[(s_1['s'] == 0) & (actual_labels['y'] == 1)].count())
    print(direct_labels[(s_1['s'] == 1) & (direct_labels['y'] == 0)].count(), actual_labels[(s_1['s'] == 1) & (actual_labels['y'] == 0)].count())
    print(direct_labels[(s_1['s'] == 1) & (direct_labels['y'] == 1)].count(), actual_labels[(s_1['s'] == 1) & (actual_labels['y'] == 1)].count())
    to_observe = DataTuple(x=feats, s=s_1, y=direct_labels, name=f"Imagined: {train.name}")

    from ethicml.visualisation.plot import save_2d_plot, save_label_plot
    save_2d_plot(to_return, './hmmm.png')
    save_2d_plot(to_observe, './hmmm_dir.png')
    save_2d_plot(train, './hmmm_og.png')

    save_label_plot(to_return, './labels_just_label_changed.png')
    save_label_plot(to_observe, './labels_all_changed.png')
    save_label_plot(train, './labels_og.png')

    pd.testing.assert_frame_equal(train.y, to_return.y)

    if post_process:
        to_return = processor.post(to_return)
        to_observe = processor.post(to_observe)

    return (
        to_return,
        TestTuple(x=pd.DataFrame(feats_test), s=test.s, name=f"Imagined: {test.name}"),
    )


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(features):
    return GradReverse.apply(features)


class FeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hid_1 = nn.Linear(2, 100)
        self.bn_1 = nn.BatchNorm1d(100)
        self.hid_2 = nn.Linear(100, 100)
        self.bn_2 = nn.BatchNorm1d(100)
        self.hid_3 = nn.Linear(100, 100)
        self.bn_3 = nn.BatchNorm1d(100)

        self.mu = nn.Linear(100, FEAT_LD)
        self.logvar = nn.Linear(100, FEAT_LD)

    def forward(self, x: torch.Tensor):
        x = self.bn_1(torch.relu(self.hid_1(x)))
        x = self.bn_2(torch.relu(self.hid_2(x)))
        x = self.bn_3(torch.relu(self.hid_3(x)))
        return td.Normal(loc=self.mu(x), scale=torch.exp(self.logvar(x)))


class FeatureDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hid_1 = nn.Linear(FEAT_LD + 1, 100)
        self.bn_1 = nn.BatchNorm1d(100)
        self.hid_2 = nn.Linear(100, 100)
        self.bn_2 = nn.BatchNorm1d(100)
        self.hid_3 = nn.Linear(100, 100)
        self.bn_3 = nn.BatchNorm1d(100)

        self.out = nn.Linear(100, 2)

    def forward(self, z: td.Distribution, s: torch.Tensor):
        x = self.bn_1(torch.relu(self.hid_1(torch.cat([z.mean, s], dim=1))))
        x = self.bn_2(torch.relu(self.hid_2(x)))
        x = self.bn_3(torch.relu(self.hid_3(x)))
        return self.out(x)


class FeatureAdv(nn.Module):
    def __init__(self):
        super().__init__()
        self.hid = nn.Linear(FEAT_LD, 100)
        self.hid_1 = nn.Linear(100, 100)
        self.bn_1 = nn.BatchNorm1d(100)
        self.hid_2 = nn.Linear(100, 100)
        self.bn_2 = nn.BatchNorm1d(100)
        self.out = nn.Linear(100, 1)

    def forward(self, z: td.Distribution):
        z = torch.relu(self.hid(grad_reverse(z.mean)))
        z = self.bn_1(torch.relu(self.hid_1(z)))
        z = self.bn_2(torch.relu(self.hid_2(z)))
        z = self.out(z)
        return td.Bernoulli(z)


class PredictionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hid_1 = nn.Linear(2, 100)
        self.bn_1 = nn.BatchNorm1d(100)
        self.hid_2 = nn.Linear(100, 100)
        self.bn_2 = nn.BatchNorm1d(100)
        self.hid_3 = nn.Linear(100, 100)
        self.bn_3 = nn.BatchNorm1d(100)
        self.mu = nn.Linear(100, PRED_LD)
        self.logvar = nn.Linear(100, PRED_LD)

    def forward(self, x: torch.Tensor):
        x = self.bn_1(torch.relu(self.hid_1(x)))
        x = self.bn_2(torch.relu(self.hid_2(x)))
        x = self.bn_3(torch.relu(self.hid_3(x)))
        return td.Bernoulli(probs=torch.sigmoid(self.mu(x)))#td.Normal(loc=self.mu(x), scale=torch.exp(self.logvar(x)))


class PredictionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hid_1 = nn.Linear(PRED_LD+1, 100)
        self.bn_1 = nn.BatchNorm1d(100)
        self.hid_2 = nn.Linear(100, 100)
        self.bn_2 = nn.BatchNorm1d(100)
        self.hid_3 = nn.Linear(100, 100)
        self.bn_3 = nn.BatchNorm1d(100)
        self.out = nn.Linear(100, 1)

    def forward(self, z: td.Distribution, s: torch.Tensor):
        x = self.bn_1(torch.relu(self.hid_1(torch.cat([z.probs, s], dim=1))))
        x = self.bn_2(torch.relu(self.hid_2(x)))
        x = self.bn_3(torch.relu(self.hid_3(x)))
        mean = self.out(x)
        return td.Bernoulli(torch.sigmoid(mean))


class DirectPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.hid = nn.Linear(3, 100)
        self.hid_1 = nn.Linear(100, 100)
        self.bn_1 = nn.BatchNorm1d(100)
        self.hid_2 = nn.Linear(100, 100)
        self.bn_2 = nn.BatchNorm1d(100)
        self.out = nn.Linear(100, 1)

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        mean = torch.relu(self.hid(torch.cat([x, s], dim=1)))
        mean = self.bn_1(torch.relu(self.hid_1(mean)))
        mean = self.bn_2(torch.relu(self.hid_2(mean)))
        mean = self.out(mean)
        return td.Bernoulli(probs=torch.sigmoid(torch.tanh(mean)))


class PredictionAdv(nn.Module):
    def __init__(self):
        super().__init__()
        self.hid = nn.Linear(PRED_LD, 100)
        self.hid_1 = nn.Linear(100, 100)
        self.bn_1 = nn.BatchNorm1d(100)
        self.hid_2 = nn.Linear(100, 100)
        self.bn_2 = nn.BatchNorm1d(100)
        self.out = nn.Linear(100, 1)

    def forward(self, z: td.Distribution):
        z = torch.relu(self.hid(grad_reverse(z.probs)))
        z = self.bn_1(torch.relu(self.hid_1(z)))
        z = self.bn_2(torch.relu(self.hid_2(z)))
        z = self.out(z)
        return td.Bernoulli(z)


class Imagine(nn.Module):
    def __init__(self, data: CustomDataset, dataset: Dataset):
        super().__init__()
        self.feature_encoder = FeatureEncoder()
        self.feature_decoder = FeatureDecoder()
        self.feature_adv = FeatureAdv()

        self.prediction_encoder = PredictionEncoder()
        self.prediction_decoder = PredictionDecoder()
        self.direct_pred = DirectPredictor()
        self.prediction_adv = PredictionAdv()

    def forward(self, x, s_1, s_2):

        feat_enc: td.Distribution = self.feature_encoder(x)
        feat_dec = self.feature_decoder(feat_enc, s_1)

        pred_enc: td.Distribution = self.prediction_encoder(x)
        pred_dec: td.Distribution = self.prediction_decoder(pred_enc, s_2)

        feat_s_pred = self.feature_adv(feat_enc)
        pred_s_pred = self.prediction_adv(pred_enc)

        direct_prediction: td.Distribution = self.direct_pred(x, s_1)

        return feat_enc, feat_dec, feat_s_pred, pred_enc, pred_dec, pred_s_pred, direct_prediction


def train_model(epoch, model, train_loader, optimizer, flags):
    """
    Train the model
    Args:
        epoch:
        model:
        train_loader:
        optimizer:
        flags:

    Returns:

    """

    model.train()
    train_loss = 0
    for batch_idx, (data_x, data_s, data_y) in enumerate(train_loader):
        data_x = data_x.to("cpu")
        data_s_1 = data_s.to("cpu")
        data_s_2 = data_s.to("cpu")
        data_y = data_y.to("cpu")

        optimizer.zero_grad()
        feat_enc: td.Distribution
        feat_dec: torch.Tensor
        feat_s_pred: td.Distribution
        pred_enc: td.Distribution
        pred_dec: td.Distribution
        pred_s_pred: td.Distribution
        direct_prediction: td.Distribution
        feat_enc, feat_dec, feat_s_pred, pred_enc, pred_dec, pred_s_pred, direct_prediction = model(data_x, data_s_1, data_s_2)

        ### Features
        recon_loss = F.mse_loss(data_x, feat_dec, reduction='mean')

        feat_prior = td.Normal(loc=torch.zeros(FEAT_LD), scale=torch.ones(FEAT_LD))
        feat_kl_loss = td.kl.kl_divergence(feat_prior, feat_enc)

        feat_sens_loss = -feat_s_pred.log_prob(data_s_1)
        ###

        ### Predictions
        pred_loss = -direct_prediction.log_prob(data_y).mean()

        pred_prior = td.Bernoulli((torch.ones_like(pred_enc.probs)/2))
        pred_kl_loss = td.kl.kl_divergence(pred_prior, pred_enc)

        pred_sens_loss = -pred_s_pred.log_prob(data_s_1)
        ###

        ### Direct Pred
        direct_loss = td.kl.kl_divergence(direct_prediction, pred_dec)
        ###

        kl_loss = feat_kl_loss.mean() + (pred_kl_loss + direct_loss).mean()
        sens_loss = (feat_sens_loss + pred_sens_loss).mean()

        loss = recon_loss + kl_loss + sens_loss + pred_loss
        loss.backward()

        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                f'train Epoch: {epoch} [{batch_idx * len(data_x)}/{len(train_loader.dataset)}'
                f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                f'Loss: {loss.item() / len(data_x):.6f}\t'
                f'recon_loss: {recon_loss.sum().item():.6f}\t'
                f'pred_loss: {pred_loss.sum().item():.6f}\t'
                f'kld_loss feats: {feat_kl_loss.sum().item():.6f}\t'
                f'kld_loss prior: {pred_kl_loss.sum().item():.6f}\t'
                f'kld_loss outps: {direct_loss.sum().item():.6f}\t'
                f'adv_feat_loss: {feat_sens_loss.sum().item():.6f}\t'
                f'adv_pred_loss: {pred_sens_loss.sum().item():.6f}\t'
            )

    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')


def main():
    """Load data from feather files, pass it to `train_and_transform` and then save the result"""
    parser = pre_algo_argparser()

    # model parameters
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--enc_size", type=int, nargs="+", required=True)
    parser.add_argument("--adv_size", type=int, nargs="+", required=True)
    parser.add_argument("--pred_size", type=int, nargs="+", required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--adv_weight", type=float, required=True)
    parser.add_argument("--validation_pcnt", type=float, required=True)
    args = parser.parse_args()
    # convert args object to a dictionary and load the feather files from the paths
    train, test = load_data_from_flags(vars(args))

    # make the argparse object type-safe (is there an easier way to do this?)
    flags = ImagineSettings(
        enc_size=args.enc_size,
        adv_size=args.adv_size,
        pred_size=args.pred_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        adv_weight=args.adv_weight,
        validation_pcnt=args.validation_pcnt,
    )
    save_transformations(train_and_transform(train, test, flags), args)


if __name__ == "__main__":
    main()
