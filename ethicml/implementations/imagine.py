import math
import os
import shutil
from itertools import groupby
from pathlib import Path
from typing import Sequence, Tuple, List

import torch
from PIL import Image
from torch.optim import Adam
from torchvision.utils import save_image
from dataclasses import dataclass
from torch import nn, optim
from torch.autograd import Function
from torch.utils.data import DataLoader
import torch.distributions as td
import torch.nn.functional as F
from tqdm import tqdm, trange, tqdm_notebook

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from ethicml.algorithms.inprocess import LR, Kamiran, LRProb, LRCV

from ethicml.data import Dataset
from ethicml.evaluators import run_metrics, metric_per_sensitive_attribute, \
    diff_per_sensitive_attribute
from ethicml.implementations.beutel import set_seed
from ethicml.implementations.pytorch_common import TestDataset, CustomDataset, RAdam
from ethicml.implementations.utils import (
    pre_algo_argparser,
    load_data_from_flags,
    save_transformations,
)
from ethicml.implementations.vfae import get_dataset_obj_by_name
from ethicml.metrics import ProbPos, Accuracy, TPR
from ethicml.preprocessing import LabelBinarizer, train_test_split
from ethicml.preprocessing.adjust_labels import assert_binary_labels
from ethicml.utility import DataTuple, TestTuple, Heaviside

_PRED_LD = 1
FEAT_LD = 50


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
    sample: int
    start_from: int
    strategy: str


def loss():
    pass


def train_and_transform(
    train: DataTuple, test: TestTuple, flags: ImagineSettings
) -> Tuple[DataTuple, TestTuple]:
    dataset = get_dataset_obj_by_name(flags.dataset)

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    set_seed(888, torch.cuda.is_available())

    post_process = False

    try:
        assert_binary_labels(train)
    except AssertionError:
        processor = LabelBinarizer()
        train = processor.adjust(train)
        post_process = True

    batch_size = 100 if flags.epochs == 0 else flags.batch_size

    # Set up the data
    _train, validate = train_test_split(train, train_percentage=0.9)
    # _train = train
    train_data = CustomDataset(_train)
    train_loader = DataLoader(train_data, batch_size=batch_size)

    valid_data = CustomDataset(validate)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)

    all_data = CustomDataset(train)
    all_data_loader = DataLoader(all_data, batch_size=100)

    test_data = CustomDataset(test)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Build Network
    current_epoch = 0
    model = Imagine(data=train_data).to(device)
    # optimizer = Adam(model.parameters(), lr=1e-3)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
    if int(flags.start_from) >= 0:
        current_epoch = int(flags.start_from)
        filename = 'checkpoint_%03d.pth.tar' % current_epoch
        PATH = Path(".") / "checkpoint" / filename
        dict_ = torch.load(PATH)
        # print(f"loaded: {dict_['epoch']}")
        model.load_state_dict(dict_['model'])
        optimizer.load_state_dict(dict_['optimizer'])
    else:
        PATH = Path(".") / "checkpoint"
        import shutil

        if PATH.exists():
            shutil.rmtree(PATH)

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=flags.start_from+1 + flags.epochs)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # Run Network
    for epoch in range(current_epoch, current_epoch + int(flags.epochs)):
        train_model(epoch, model, train_loader, valid_loader, optimizer, device, flags)
        scheduler.step(epoch)

    # PATH = Path(".") / "checkpoint" / 'model_best.pth.tar'
    # dict_ = torch.load(PATH)
    # print(f"Best model was at step: {dict_['epoch']}")
    # model.load_state_dict(dict_['model'])
    # model.eval()

    # Transform output
    actual_feats_train: pd.DataFrame = pd.DataFrame(columns=_train.x.columns)
    feats_train_encs: pd.DataFrame = pd.DataFrame(columns=list(range(FEAT_LD)))
    feats_train_encs_og_only: pd.DataFrame = pd.DataFrame(columns=list(range(FEAT_LD)))
    feats_train: pd.DataFrame = pd.DataFrame(columns=[col for col in _train.x.columns]+["id"])
    s_1_list_train: pd.DataFrame = pd.DataFrame(columns=[col for col in _train.s.columns]+["id"])
    s_2_list_train: pd.DataFrame = pd.DataFrame(columns=[col for col in _train.s.columns]+["id"])
    actual_labels_train: pd.DataFrame = pd.DataFrame(columns=[col for col in _train.y.columns]+["id"])
    actual_sens_train: pd.DataFrame = pd.DataFrame(columns=[col for col in _train.s.columns]+["id"])
    direct_preds_train: pd.DataFrame = pd.DataFrame(columns=_train.y.columns)
    preds_train_encs: pd.DataFrame = pd.DataFrame(columns=_train.y.columns)
    preds_train: pd.DataFrame = pd.DataFrame(columns=[col for col in _train.y.columns]+["id"])

    actual_feats_test: pd.DataFrame = pd.DataFrame(columns=_train.x.columns)
    feats_test_encs: pd.DataFrame = pd.DataFrame(columns=list(range(FEAT_LD)))
    feats_test: pd.DataFrame = pd.DataFrame(columns=[col for col in _train.x.columns] + ["id"])
    s_1_list_test: pd.DataFrame = pd.DataFrame(columns=[col for col in _train.s.columns] + ["id"])
    s_2_list_test: pd.DataFrame = pd.DataFrame(columns=[col for col in _train.s.columns] + ["id"])
    actual_labels_test: pd.DataFrame = pd.DataFrame(columns=[col for col in _train.y.columns] + ["id"])
    actual_sens_test: pd.DataFrame = pd.DataFrame(columns=[col for col in _train.s.columns] + ["id"])
    direct_preds_test: pd.DataFrame = pd.DataFrame(columns=_train.y.columns)
    preds_test_encs: pd.DataFrame = pd.DataFrame(columns=_train.y.columns)
    preds_test: pd.DataFrame = pd.DataFrame(columns=[col for col in _train.y.columns] + ["id"])

    SAMPLES = flags.sample

    first = True
    first_flip_x = True
    to_plot = None

    model.eval()
    with torch.no_grad():
        for _i, _x, _s, _y, _out in all_data_loader:

            _i = _i.to(device)
            _x = _x.to(device)
            _s = _s.to(device)
            _y = _y.to(device)
            _out = [out.to(device) for out in _out]

            ###
            # original data
            ###
            for samp in range(SAMPLES):
                _s_1 = _s.detach().clone()
                _s_2 = _s.detach().clone()
                feat_enc, feat_dec, feat_dec_flip, feat_s_pred, pred_enc, pred_dec, pred_dec_flip, pred_s_pred, direct_prediction = model(
                    _x, _s_1, _s_2, _s
                )

                if first:
                    save_image(_x, Path(f"./original_1_sample_{samp}_{flags.strategy}.png"))
                    to_plot = _x.cpu()
                    rec = torch.cat([feat for feat in feat_dec], 1).cpu()
                    save_image(rec, Path(f"./recon_x_sample_{samp}_{flags.strategy}.png"))
                    diff = to_plot - rec
                    save_image(diff, Path(f"./difference_in_recon_sample_{samp}_{flags.strategy}.png"), normalize=True)

                # for _ in range(SAMPLES):
                feats_train = pd.concat(
                    [
                        feats_train,
                        pd.concat([
                            pd.DataFrame(_x.cpu().numpy(), columns=_train.x.columns),
                            pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                        ], axis='columns', ignore_index=False)
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                feats_train_encs_og_only = pd.concat(
                    [
                        feats_train_encs_og_only,
                        pd.DataFrame(feat_enc.mean.cpu().numpy(), columns=list(range(FEAT_LD))),
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                feats_train_encs = pd.concat(
                    [
                        feats_train_encs,
                        pd.DataFrame(feat_enc.mean.cpu().numpy(), columns=list(range(FEAT_LD))),
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                direct_preds_train = pd.concat(
                    [
                        direct_preds_train,
                        pd.DataFrame(
                            direct_prediction.sigmoid().cpu().numpy(), columns=_train.y.columns
                        ),
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                preds_train = pd.concat(
                    [
                        preds_train,
                        pd.concat([
                            pd.DataFrame(_y.cpu().numpy(), columns=_train.y.columns, dtype=np.int64),
                            pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                        ], axis='columns', ignore_index=False)
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                s_1_list_train = pd.concat(
                    [
                        s_1_list_train,
                        pd.concat([
                            pd.DataFrame(_s_1.cpu().numpy(), columns=_train.s.columns, dtype=np.int64),
                            pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                        ], axis='columns', ignore_index=False)
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                s_2_list_train = pd.concat(
                    [
                        s_2_list_train,
                        pd.concat([
                            pd.DataFrame(_s_2.cpu().numpy(), columns=_train.s.columns, dtype=np.int64),
                            pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                        ], axis='columns', ignore_index=False)
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                actual_labels_train = pd.concat(
                    [
                        actual_labels_train,
                        pd.concat([
                            pd.DataFrame(_y.cpu().numpy(), columns=_train.y.columns, dtype=np.int64),
                            pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                        ], axis='columns', ignore_index=False)
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                actual_sens_train = pd.concat(
                    [
                        actual_sens_train,
                        pd.concat([
                            pd.DataFrame(_s.cpu().numpy(), columns=_train.s.columns,
                                         dtype=np.int64),
                            pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                        ], axis='columns', ignore_index=False)
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                preds_train_encs = pd.concat(
                    [
                        preds_train_encs,
                        pd.DataFrame(pred_enc.sigmoid().cpu().numpy(), columns=_train.y.columns),
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                actual_feats_train = pd.concat(
                    [actual_feats_train, pd.DataFrame(_x.cpu().numpy(), columns=_train.x.columns)],
                    axis='rows',
                    ignore_index=True,
                )

                ###
                # flippedx, og y
                ###
                if flags.strategy in ["flip_x", "use_all"]:
                    _s_1 = (_s.detach().clone() - 1) ** 2
                    _s_2 = _s.detach().clone()
                    # feat_enc, feat_dec, feat_dec_flip, feat_s_pred, pred_enc, pred_dec, pred_dec_flip, pred_s_pred, direct_prediction = model(
                    #     _x, _s_1, _s_2, _s.to(device)
                    # )

                    if first_flip_x:
                        save_image(_x, Path(f"./original_sample_{samp}_{flags.strategy}.png"))
                        to_plot = _x.cpu()
                        rec_flip = torch.cat([feat for feat in feat_dec], 1).cpu()
                        rec_og = torch.cat([feat for feat in feat_dec], 1).cpu()
                        rec_fl = torch.cat([feat for feat in feat_dec_flip], 1).cpu()
                        save_image(rec_flip, Path(f"./flipped_x_sample_{samp}_{flags.strategy}.png"))
                        diff = to_plot - rec_flip
                        save_image(diff, Path(f"./difference_sample_{samp}_{flags.strategy}.png"), normalize=True)
                        save_image(rec - rec_flip, Path(f"./difference_rec_to_flip_sample{samp}_{flags.strategy}.png"), normalize=True)

                        fig, ax = plt.subplots(figsize=(10,14))
                        im = ax.imshow(diff.numpy())

                        # We want to show all ticks...
                        ax.set_xticks(np.arange(len(train.x.columns)))

                        # ax.set_yticks(np.arange(diff.shape[1]))
                        # ... and label them with the respective list entries
                        ax.set_xticklabels(train.x.columns)
                        # ax.set_yticklabels(vegetables)

                        ax.tick_params(top=True, bottom=False,
                                       labeltop=True, labelbottom=False)

                        # Rotate the tick labels and set their alignment.
                        plt.setp(ax.get_xticklabels(), rotation=90, ha="left",
                                 rotation_mode="anchor")

                        # Loop over data dimensions and create text annotations.
                        # for i in range(len(vegetables)):
                        #     for j in range(len(farmers)):
                        #         text = ax.text(j, i, harvest[i, j],
                        #                        ha="center", va="center", color="w")

                        ax.set_title("Difference between Gender reconstructions")
                        fig.tight_layout()
                        plt.show()
                        fig.savefig(f'./difference_sex_sample_{samp}_{flags.strategy}.png')



                        fig_1, ax_1 = plt.subplots(figsize=(10, 14))
                        im = ax_1.imshow((rec_og - rec_fl).numpy())

                        # We want to show all ticks...
                        ax_1.set_xticks(np.arange(len(train.x.columns)))

                        # ax.set_yticks(np.arange(diff.shape[1]))
                        # ... and label them with the respective list entries
                        ax_1.set_xticklabels(train.x.columns)
                        # ax.set_yticklabels(vegetables)

                        ax_1.tick_params(top=True, bottom=False,
                                       labeltop=True, labelbottom=False)

                        # Rotate the tick labels and set their alignment.
                        plt.setp(ax_1.get_xticklabels(), rotation=90, ha="left",
                                 rotation_mode="anchor")

                        # Loop over data dimensions and create text annotations.
                        # for i in range(len(vegetables)):
                        #     for j in range(len(farmers)):
                        #         text = ax.text(j, i, harvest[i, j],
                        #                        ha="center", va="center", color="w")

                        ax_1.set_title("Difference between Gender reconstructions recon x")
                        fig_1.tight_layout()
                        plt.show()
                        fig_1.savefig(f'./difference_sex_recon_sample_{samp}_{flags.strategy}.png')

                    # for _ in range(SAMPLES):
                    feats_train = pd.concat(
                        [
                            feats_train,
                            pd.concat([
                                pd.DataFrame(torch.cat([feat for feat in feat_dec_flip], 1).cpu().numpy(), columns=_train.x.columns),
                                pd.DataFrame(_i.cpu().numpy(), columns=["id"])], axis='columns', ignore_index=False
                            )
                        ],
                        axis='rows',
                        ignore_index=True,
                    )
                    feats_train_encs = pd.concat(
                        [
                            feats_train_encs,
                            pd.DataFrame(feat_enc.mean.cpu().numpy(), columns=list(range(FEAT_LD))),
                        ],
                        axis='rows',
                        ignore_index=True,
                    )
                    direct_preds_train = pd.concat(
                        [
                            direct_preds_train,
                            pd.DataFrame(
                                direct_prediction.sigmoid().cpu().numpy(), columns=_train.y.columns
                            ),
                        ],
                        axis='rows',
                        ignore_index=True,
                    )
                    preds_train = pd.concat(
                        [
                            preds_train,
                            pd.concat([
                                pd.DataFrame(_y.cpu().numpy(), columns=_train.y.columns,
                                             dtype=np.int64),
                                pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                            ], axis='columns', ignore_index=False)
                        ],
                        axis='rows',
                        ignore_index=True,
                    )
                    s_1_list_train = pd.concat(
                        [
                            s_1_list_train,
                            pd.concat([
                                pd.DataFrame(_s_1.cpu().numpy(), columns=_train.s.columns,
                                             dtype=np.int64),
                                pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                            ], axis='columns', ignore_index=False)
                        ],
                        axis='rows',
                        ignore_index=True,
                    )
                    s_2_list_train = pd.concat(
                        [
                            s_2_list_train,
                            pd.concat([
                                pd.DataFrame(_s_2.cpu().numpy(), columns=_train.s.columns,
                                             dtype=np.int64),
                                pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                            ], axis='columns', ignore_index=False)
                        ],
                        axis='rows',
                        ignore_index=True,
                    )
                    actual_labels_train = pd.concat(
                        [
                            actual_labels_train,
                            pd.concat([
                                pd.DataFrame(_y.cpu().numpy(), columns=_train.y.columns, dtype=np.int64),
                                pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                            ], axis='columns', ignore_index=False)
                        ],
                        axis='rows',
                        ignore_index=True,
                    )
                    actual_sens_train = pd.concat(
                        [
                            actual_sens_train,
                            pd.concat([
                                pd.DataFrame(_s.cpu().numpy(), columns=_train.s.columns,
                                             dtype=np.int64),
                                pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                            ], axis='columns', ignore_index=False)
                        ],
                        axis='rows',
                        ignore_index=True,
                    )
                    actual_feats_train = pd.concat(
                        [actual_feats_train, pd.DataFrame(_x.cpu().numpy(), columns=_train.x.columns)],
                        axis='rows',
                        ignore_index=True,
                    )

                ###
                # flipped x, flipped y
                ###
                if flags.strategy in ["flip_both"]:
                    _s_1 = (_s.detach().clone() - 1) ** 2
                    _s_2 = (_s.detach().clone() - 1) ** 2
                    # feat_enc, feat_dec_2, feat_dec_flip, feat_s_pred, pred_enc, pred_dec, pred_dec_flip, pred_s_pred, direct_prediction = model(
                    #     torch.cat([feat for feat in feat_dec], 1), _s_1, _s_2, _s.to(device)
                    # )
                    # for _ in range(SAMPLES):
                    feats_train = pd.concat(
                        [
                            feats_train,
                            pd.concat([
                                pd.DataFrame(
                                    torch.cat([feat for feat in feat_dec_flip],
                                              1).cpu().numpy(),
                                    columns=_train.x.columns,
                                ),
                                pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                            ], axis='columns', ignore_index=False)
                        ],
                        axis='rows',
                        ignore_index=True,
                    )
                    feats_train_encs = pd.concat(
                        [
                            feats_train_encs,
                            pd.DataFrame(feat_enc.mean.cpu().numpy(), columns=list(range(FEAT_LD))),
                        ],
                        axis='rows',
                        ignore_index=True,
                    )
                    direct_preds_train = pd.concat(
                        [
                            direct_preds_train,
                            pd.DataFrame(
                                direct_prediction.sigmoid().cpu().numpy(), columns=_train.y.columns
                            ),
                        ],
                        axis='rows',
                        ignore_index=True,
                    )
                    preds_train = pd.concat(
                        [
                            preds_train,
                            pd.concat([
                                pd.DataFrame(pred_dec_flip.sigmoid().cpu().numpy(), columns=_train.y.columns),
                                pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                            ], axis='columns', ignore_index=False)
                        ],
                        axis='rows',
                        ignore_index=True,
                    )
                    s_1_list_train = pd.concat(
                        [
                            s_1_list_train,
                            pd.concat([
                                pd.DataFrame(_s_1.cpu().numpy(), columns=_train.s.columns,
                                             dtype=np.int64),
                                pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                            ], axis='columns', ignore_index=False)
                        ],
                        axis='rows',
                        ignore_index=True,
                    )
                    s_2_list_train = pd.concat(
                        [
                            s_2_list_train,
                            pd.concat([
                                pd.DataFrame(_s_2.cpu().numpy(), columns=_train.s.columns,
                                             dtype=np.int64),
                                pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                            ], axis='columns', ignore_index=False)
                        ],
                        axis='rows',
                        ignore_index=True,
                    )
                    actual_labels_train = pd.concat(
                        [
                            actual_labels_train,
                            pd.concat([
                                pd.DataFrame(_y.cpu().numpy(), columns=_train.y.columns,
                                             dtype=np.int64),
                                pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                            ], axis='columns', ignore_index=False)
                        ],
                        axis='rows',
                        ignore_index=True,
                    )
                    actual_sens_train = pd.concat(
                        [
                            actual_sens_train,
                            pd.concat([
                                pd.DataFrame(_s.cpu().numpy(), columns=_train.s.columns,
                                             dtype=np.int64),
                                pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                            ], axis='columns', ignore_index=False)
                        ],
                        axis='rows',
                        ignore_index=True,
                    )
                    actual_feats_train = pd.concat(
                        [actual_feats_train, pd.DataFrame(_x.cpu().numpy(), columns=_train.x.columns)],
                        axis='rows',
                        ignore_index=True,
                    )

                ###
                # og x, flipped y
                ###
                if flags.strategy in ["flip_y", "use_all"]:
                    _s_1 = _s.detach().clone()
                    _s_2 = (_s.detach().clone() - 1) ** 2
                    # feat_enc, feat_dec, feat_dec_flip, feat_s_pred, pred_enc, pred_dec, pred_dec_flip, pred_s_pred, direct_prediction = model(
                    #     _x, _s_1, _s_2, _s.to(device)
                    # )
                    # for _ in range(SAMPLES):
                    feats_train = pd.concat(
                        [
                            feats_train,
                            pd.concat([
                                pd.DataFrame(_x.cpu().numpy(), columns=_train.x.columns),
                                pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                            ], axis='columns', ignore_index=False)
                        ],
                        axis='rows',
                        ignore_index=True,
                    )
                    feats_train_encs = pd.concat(
                        [
                            feats_train_encs,
                            pd.DataFrame(feat_enc.mean.cpu().numpy(), columns=list(range(FEAT_LD))),
                        ],
                        axis='rows',
                        ignore_index=True,
                    )
                    direct_preds_train = pd.concat(
                        [
                            direct_preds_train,
                            pd.DataFrame(
                                direct_prediction.sigmoid().cpu().numpy(), columns=_train.y.columns
                            ),
                        ],
                        axis='rows',
                        ignore_index=True,
                    )
                    preds_train = pd.concat(
                        [
                            preds_train,
                            pd.concat([
                                pd.DataFrame(pred_dec_flip.sigmoid().cpu().numpy(),
                                             columns=_train.y.columns),
                                pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                            ], axis='columns', ignore_index=False)
                        ],
                        axis='rows',
                        ignore_index=True,
                    )
                    s_1_list_train = pd.concat(
                        [
                            s_1_list_train,
                            pd.concat([
                                pd.DataFrame(_s_1.cpu().numpy(), columns=_train.s.columns,
                                             dtype=np.int64),
                                pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                            ], axis='columns', ignore_index=False)
                        ],
                        axis='rows',
                        ignore_index=True,
                    )
                    s_2_list_train = pd.concat(
                        [
                            s_2_list_train,
                            pd.concat([
                                pd.DataFrame(_s_2.cpu().numpy(), columns=_train.s.columns,
                                             dtype=np.int64),
                                pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                            ], axis='columns', ignore_index=False)
                        ],
                        axis='rows',
                        ignore_index=True,
                    )
                    actual_labels_train = pd.concat(
                        [
                            actual_labels_train,
                            pd.concat([
                                pd.DataFrame(_y.cpu().numpy(), columns=_train.y.columns,
                                             dtype=np.int64),
                                pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                            ], axis='columns', ignore_index=False)
                        ],
                        axis='rows',
                        ignore_index=True,
                    )
                    actual_sens_train = pd.concat(
                        [
                            actual_sens_train,
                            pd.concat([
                                pd.DataFrame(_s.cpu().numpy(), columns=_train.s.columns,
                                             dtype=np.int64),
                                pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                            ], axis='columns', ignore_index=False)
                        ],
                        axis='rows',
                        ignore_index=True,
                    )
                    actual_feats_train = pd.concat(
                        [actual_feats_train, pd.DataFrame(_x.cpu().numpy(), columns=_train.x.columns)],
                        axis='rows',
                        ignore_index=True,
                    )

            first = False
            first_flip_x = False

        for _i, _x, _s, _y, _out in test_loader:
            _i = _i.to(device)
            _x = _x.to(device)
            _s = _s.to(device)
            _y = _y.to(device)
            _out = [out.to(device) for out in _out]

            ###
            # original data
            ###
            # for _ in range(SAMPLES):
            _s_1 = _s.detach().clone()
            _s_2 = _s.detach().clone()
            feat_enc, feat_dec, feat_dec_flip, feat_s_pred, pred_enc, pred_dec, pred_dec_flip, pred_s_pred, direct_prediction = model(_x, _s_1, _s_2, _s.to(device))
            feats_test = pd.concat(
                [
                    feats_test,
                    pd.concat([
                        pd.DataFrame(_x.cpu().numpy(), columns=_train.x.columns),
                        pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                    ], axis='columns', ignore_index=False)
                ],
                axis='rows',
                ignore_index=True,
            )
            feats_test_encs = pd.concat(
                [
                    feats_test_encs,
                    pd.DataFrame(feat_enc.mean.cpu().numpy(), columns=list(range(FEAT_LD))),
                ],
                axis='rows',
                ignore_index=True,
            )
            direct_preds_test = pd.concat(
                [
                    direct_preds_test,
                    pd.DataFrame(
                        direct_prediction.sigmoid().cpu().numpy(), columns=_train.y.columns
                    ),
                ],
                axis='rows',
                ignore_index=True,
            )
            preds_test = pd.concat(
                [
                    preds_test,
                    pd.concat([
                        pd.DataFrame(_y.cpu().numpy(), columns=_train.y.columns,
                                     dtype=np.int64),
                        pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                    ], axis='columns', ignore_index=False)
                ],
                axis='rows',
                ignore_index=True,
            )
            s_1_list_test = pd.concat(
                [
                    s_1_list_test,
                    pd.concat([
                        pd.DataFrame(_s_1.cpu().numpy(), columns=_train.s.columns,
                                     dtype=np.int64),
                        pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                    ], axis='columns', ignore_index=False)
                ],
                axis='rows',
                ignore_index=True,
            )
            s_2_list_test = pd.concat(
                [
                    s_2_list_test,
                    pd.concat([
                        pd.DataFrame(_s_2.cpu().numpy(), columns=_train.s.columns,
                                     dtype=np.int64),
                        pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                    ], axis='columns', ignore_index=False)
                ],
                axis='rows',
                ignore_index=True,
            )
            actual_labels_test = pd.concat(
                [
                    actual_labels_test,
                    pd.concat([
                        pd.DataFrame(_y.cpu().numpy(), columns=_train.y.columns,
                                     dtype=np.int64),
                        pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                    ], axis='columns', ignore_index=False)
                ],
                axis='rows',
                ignore_index=True,
            )
            actual_sens_test = pd.concat(
                [
                    actual_sens_test,
                    pd.concat([
                        pd.DataFrame(_s.cpu().numpy(), columns=_train.s.columns,
                                     dtype=np.int64),
                        pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                    ], axis='columns', ignore_index=False)
                ],
                axis='rows',
                ignore_index=True,
            )
            preds_test_encs = pd.concat(
                [
                    preds_test_encs,
                    pd.DataFrame(pred_enc.sigmoid().cpu().numpy(), columns=_train.y.columns),
                ],
                axis='rows',
                ignore_index=True,
            )
            actual_feats_test = pd.concat(
                [actual_feats_test, pd.DataFrame(_x.cpu().numpy(), columns=_train.x.columns)],
                axis='rows',
                ignore_index=True,
            )

            ###
            # flippedx, og y
            ###
            if flags.strategy in ["flip_x", "use_all"]:
                _s_1 = (_s.detach().clone() - 1) ** 2
                _s_2 = _s.detach().clone()
                # feat_enc, feat_dec, feat_dec_flip, feat_s_pred, pred_enc, pred_dec, pred_dec_flip, pred_s_pred, direct_prediction = model(
                #     _x, _s_1, _s_2, _s.to(device)
                # )
                # for _ in range(SAMPLES):
                feats_test = pd.concat(
                    [
                        feats_test,
                        pd.concat([
                            pd.DataFrame(torch.cat([feat for feat in feat_dec_flip],
                                                   1).cpu().numpy(), columns=_train.x.columns),
                            pd.DataFrame(_i.cpu().numpy(), columns=["id"])], axis='columns',
                            ignore_index=False
                        )
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                feats_test_encs = pd.concat(
                    [
                        feats_test_encs,
                        pd.DataFrame(feat_enc.mean.cpu().numpy(), columns=list(range(FEAT_LD))),
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                direct_preds_test = pd.concat(
                    [
                        direct_preds_test,
                        pd.DataFrame(
                            direct_prediction.sigmoid().cpu().numpy(), columns=_train.y.columns
                        ),
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                preds_test = pd.concat(
                    [
                        preds_test,
                        pd.concat([
                            pd.DataFrame(_y.cpu().numpy(), columns=_train.y.columns,
                                         dtype=np.int64),
                            pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                        ], axis='columns', ignore_index=False)
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                s_1_list_test = pd.concat(
                    [
                        s_1_list_test,
                        pd.concat([
                            pd.DataFrame(_s_1.cpu().numpy(), columns=_train.s.columns,
                                         dtype=np.int64),
                            pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                        ], axis='columns', ignore_index=False)
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                s_2_list_test = pd.concat(
                    [
                        s_2_list_test,
                        pd.concat([
                            pd.DataFrame(_s_2.cpu().numpy(), columns=_train.s.columns,
                                         dtype=np.int64),
                            pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                        ], axis='columns', ignore_index=False)
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                actual_labels_test = pd.concat(
                    [
                        actual_labels_test,
                        pd.concat([
                            pd.DataFrame(_y.cpu().numpy(), columns=_train.y.columns,
                                         dtype=np.int64),
                            pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                        ], axis='columns', ignore_index=False)
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                actual_sens_test = pd.concat(
                    [
                        actual_sens_test,
                        pd.concat([
                            pd.DataFrame(_s.cpu().numpy(), columns=_train.s.columns,
                                         dtype=np.int64),
                            pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                        ], axis='columns', ignore_index=False)
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                actual_feats_test = pd.concat(
                    [actual_feats_test,
                     pd.DataFrame(_x.cpu().numpy(), columns=_train.x.columns)],
                    axis='rows',
                    ignore_index=True,
                )

            ###
            # flipped x, flipped y
            ###
            if flags.strategy in ["flip_both"]:
                _s_1 = (_s.detach().clone() - 1) ** 2
                _s_2 = (_s.detach().clone() - 1) ** 2
                # feat_enc, feat_dec, feat_dec_flip, feat_s_pred, pred_enc, pred_dec, pred_dec_flip, pred_s_pred, direct_prediction = model(
                #     torch.cat([feat for feat in feat_dec], 1), _s_1, _s_2, _s.to(device)
                # )
                # for _ in range(SAMPLES):
                feats_test = pd.concat(
                    [
                        feats_test,
                        pd.concat([
                            pd.DataFrame(
                                torch.cat([feat for feat in feat_dec_flip],
                                          1).cpu().numpy(),
                                columns=_train.x.columns,
                            ),
                            pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                        ], axis='columns', ignore_index=False)
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                feats_test_encs = pd.concat(
                    [
                        feats_test_encs,
                        pd.DataFrame(feat_enc.mean.cpu().numpy(), columns=list(range(FEAT_LD))),
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                direct_preds_test = pd.concat(
                    [
                        direct_preds_test,
                        pd.DataFrame(
                            direct_prediction.sigmoid().cpu().numpy(), columns=_train.y.columns
                        ),
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                preds_test = pd.concat(
                    [
                        preds_test,
                        pd.concat([
                            pd.DataFrame(pred_dec_flip.sigmoid().cpu().numpy(),
                                         columns=_train.y.columns),
                            pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                        ], axis='columns', ignore_index=False)
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                s_1_list_test = pd.concat(
                    [
                        s_1_list_test,
                        pd.concat([
                            pd.DataFrame(_s_1.cpu().numpy(), columns=_train.s.columns,
                                         dtype=np.int64),
                            pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                        ], axis='columns', ignore_index=False)
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                s_2_list_test = pd.concat(
                    [
                        s_2_list_test,
                        pd.concat([
                            pd.DataFrame(_s_2.cpu().numpy(), columns=_train.s.columns,
                                         dtype=np.int64),
                            pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                        ], axis='columns', ignore_index=False)
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                actual_labels_test = pd.concat(
                    [
                        actual_labels_test,
                        pd.concat([
                            pd.DataFrame(_y.cpu().numpy(), columns=_train.y.columns,
                                         dtype=np.int64),
                            pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                        ], axis='columns', ignore_index=False)
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                actual_sens_test = pd.concat(
                    [
                        actual_sens_test,
                        pd.concat([
                            pd.DataFrame(_s.cpu().numpy(), columns=_train.s.columns,
                                         dtype=np.int64),
                            pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                        ], axis='columns', ignore_index=False)
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                actual_feats_test = pd.concat(
                    [actual_feats_test,
                     pd.DataFrame(_x.cpu().numpy(), columns=_train.x.columns)],
                    axis='rows',
                    ignore_index=True,
                )

            ###
            # og x, flipped y
            ###
            if flags.strategy in ["flip_y", "use_all"]:
                _s_1 = _s.detach().clone()
                _s_2 = (_s.detach().clone() - 1) ** 2
                # feat_enc, feat_dec, feat_dec_flip, feat_s_pred, pred_enc, pred_dec, pred_dec_flip, pred_s_pred, direct_prediction = model(
                #     _x, _s_1, _s_2, _s.to(device)
                # )
                # for _ in range(SAMPLES):
                feats_test = pd.concat(
                    [
                        feats_test,
                        pd.concat([
                            pd.DataFrame(_x.cpu().numpy(), columns=_train.x.columns),
                            pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                        ], axis='columns', ignore_index=False)
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                feats_test_encs = pd.concat(
                    [
                        feats_test_encs,
                        pd.DataFrame(feat_enc.mean.cpu().numpy(), columns=list(range(FEAT_LD))),
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                direct_preds_test = pd.concat(
                    [
                        direct_preds_test,
                        pd.DataFrame(
                            direct_prediction.sigmoid().cpu().numpy(), columns=_train.y.columns
                        ),
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                preds_test = pd.concat(
                    [
                        preds_test,
                        pd.concat([
                            pd.DataFrame(pred_dec_flip.sigmoid().cpu().numpy(),
                                         columns=_train.y.columns),
                            pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                        ], axis='columns', ignore_index=False)
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                s_1_list_test = pd.concat(
                    [
                        s_1_list_test,
                        pd.concat([
                            pd.DataFrame(_s_1.cpu().numpy(), columns=_train.s.columns,
                                         dtype=np.int64),
                            pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                        ], axis='columns', ignore_index=False)
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                s_2_list_test = pd.concat(
                    [
                        s_2_list_test,
                        pd.concat([
                            pd.DataFrame(_s_2.cpu().numpy(), columns=_train.s.columns,
                                         dtype=np.int64),
                            pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                        ], axis='columns', ignore_index=False)
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                actual_labels_test = pd.concat(
                    [
                        actual_labels_test,
                        pd.concat([
                            pd.DataFrame(_y.cpu().numpy(), columns=_train.y.columns,
                                         dtype=np.int64),
                            pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                        ], axis='columns', ignore_index=False)
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                actual_sens_test = pd.concat(
                    [
                        actual_sens_test,
                        pd.concat([
                            pd.DataFrame(_s.cpu().numpy(), columns=_train.s.columns,
                                         dtype=np.int64),
                            pd.DataFrame(_i.cpu().numpy(), columns=["id"])
                        ], axis='columns', ignore_index=False)
                    ],
                    axis='rows',
                    ignore_index=True,
                )
                actual_feats_test = pd.concat(
                    [actual_feats_test,
                     pd.DataFrame(_x.cpu().numpy(), columns=_train.x.columns)],
                    axis='rows',
                    ignore_index=True,
                )

        direct_preds_train = direct_preds_train.applymap(lambda x: 1 if x >= 0.5 else 0)

        preds_train[train.y.columns[0]] = preds_train[train.y.columns[0]].map(lambda x: 1 if x>= 0.5 else 0)
        preds_test[train.y.columns[0]] = preds_test[train.y.columns[0]].map(lambda x: 1 if x>= 0.5 else 0)

        actual_labels_train[train.y.columns[0]] = actual_labels_train[train.y.columns[0]].map(lambda x: 1 if x>= 0.5 else 0)
        actual_sens_train[train.s.columns[0]] = actual_sens_train[train.s.columns[0]].map(lambda x: 1 if x>= 0.5 else 0)
        actual_labels_test[train.y.columns[0]] = actual_labels_test[train.y.columns[0]].map(lambda x: 1 if x>= 0.5 else 0)
        actual_sens_test[train.s.columns[0]] = actual_sens_test[train.s.columns[0]].map(lambda x: 1 if x>= 0.5 else 0)

        print("=============================")
        print(f"strategy: {flags.strategy}")

        _mod = LRCV()
        clf = _mod.fit(
            DataTuple(x=feats_train.drop("id", axis=1), s=s_1_list_train.drop('id', axis=1),
                      y=preds_train.drop('id', axis=1)))
        total = 0
        tpr_total = 0
        _total = 0
        acc = 0
        tpr_count = 0
        feats_grouped = feats_test.groupby("id")
        labels_grouped = preds_test.groupby("id")
        sens_grouped = s_1_list_test.groupby("id")
        actual_grouped = actual_labels_test.groupby("id")

        s_col = train.s.columns[0]
        y_col = train.y.columns[0]

        preds_im = pd.DataFrame(clf.predict(test.x), columns=['preds'])
        tqdm.write(f"im train acc: {Accuracy().score(preds_im, test)}")

        res = metric_per_sensitive_attribute(preds_im, test, ProbPos())
        tqdm.write(f"im train dp: {res}")
        tqdm.write(f"im train dp: {diff_per_sensitive_attribute(res).values()}")

        res = metric_per_sensitive_attribute(preds_im, test, TPR())
        tqdm.write(f"im train eq op: {res}")
        tqdm.write(f"im train eq op: {diff_per_sensitive_attribute(res).values()}")

        print(f"P( y^=1 ) = {preds_im['preds'].mean()}")
        print(f"P( y^=1 | y=1 ) = {preds_im['preds'][test.y[y_col] == 1].mean()}")
        print(f"P( y^=1 | s=1 ) = {preds_im['preds'][test.s[s_col] == 1].mean()}")
        print(f"P( y^=1 | s=1, y=1 ) = {preds_im['preds'][(test.s[s_col] == 1) & (test.y[y_col] == 1)].mean()}")
        print(f"P( y^=1 | s=0 ) = {preds_im['preds'][test.s[s_col] == 0].mean()}")
        print(f"P( y^=1 | s=0, y=1 ) = {preds_im['preds'][(test.s[s_col] == 0) & (test.y[y_col] == 1)].mean()}")

        print(f"P( y=1 ) = {test.y[y_col].mean()}")
        print(f"P( y=1 | s=1 ) = {test.y[y_col][test.s[s_col] == 1].mean()}")
        print(f"P( y=1 | s=1, y=1 ) = {test.y[y_col][(test.s[s_col] == 1) & (test.y[y_col])].mean()}")
        print(f"P( y=1 | s=0 ) = {test.y[y_col][test.s[s_col] == 0].mean()}")
        print(f"P( y=1 | s=0, y=1 ) = {test.y[y_col][(test.s[s_col] == 0) & (test.y[y_col])].mean()}")

        if not flags.strategy == "flip_y":
            for l, ((_, x_g), (_, s_g), (_, y_g), (_, a_g)) in enumerate(
                tqdm_notebook(zip(feats_grouped, sens_grouped, labels_grouped, actual_grouped),
                     total=test.x.shape[0], desc="im ind par")):
                _mod_preds = pd.DataFrame(clf.predict(x_g.drop(["id"], axis=1)), columns=["preds"])
                tt = DataTuple(x=x_g.drop(["id"], axis=1).reset_index(drop=True),
                               s=s_g.drop(["id"], axis=1).reset_index(drop=True),
                               y=a_g.drop(["id"], axis=1).reset_index(drop=True),
                               name=f"Imagined: {train.name}")
                res = metric_per_sensitive_attribute(_mod_preds, tt, ProbPos())
                total += sum(list(diff_per_sensitive_attribute(res).values()))

                if a_g[a_g.columns[0]].values[0] == 1:
                    tpr_total += sum(list(diff_per_sensitive_attribute(res).values()))
                    tpr_count += 1

                _total += (_mod_preds.values[0] ^ _mod_preds.values[1]).sum()
                acc += 2 - abs(_mod_preds.values - (a_g.drop("id", axis=1).values)).sum()
            tqdm.write(f"Im Ind. Parity {_mod.name}: {total / (test.x.shape[0])}")
            tqdm.write(f"Im Ind. Eq Opp {_mod.name}: {tpr_total / tpr_count}")



        _mod = Kamiran()
        clf = _mod.fit(train)
        total = 0
        tpr_total = 0
        _total = 0
        acc = 0
        tpr_count = 0
        feats_grouped = feats_test.groupby("id")
        labels_grouped = preds_test.groupby("id")
        sens_grouped = s_1_list_test.groupby("id")
        actual_grouped = actual_labels_test.groupby("id")

        preds_kc = pd.DataFrame(clf.predict(test.x), columns=['preds'])
        tqdm.write(f"kc train acc: {Accuracy().score(preds_kc, test)}")

        res = metric_per_sensitive_attribute(preds_kc, test, ProbPos())
        tqdm.write(f"kc train dp: {res}")
        tqdm.write(f"kc train dp: {diff_per_sensitive_attribute(res).values()}")

        res = metric_per_sensitive_attribute(preds_kc, test, TPR())
        tqdm.write(f"kc train eq op: {res}")
        tqdm.write(f"kc train eq op: {diff_per_sensitive_attribute(res).values()}")

        print(f"P( y^=1 ) = {preds_kc['preds'].mean()}")
        print(f"P( y^=1 | y=1 ) = {preds_kc['preds'][test.y[y_col] == 1].mean()}")
        print(f"P( y^=1 | s=1 ) = {preds_kc['preds'][test.s[s_col] == 1].mean()}")
        print(
            f"P( y^=1 | s=1, y=1 ) = {preds_kc['preds'][(test.s[s_col] == 1) & (test.y[y_col] == 1)].mean()}")
        print(f"P( y^=1 | s=0 ) = {preds_kc['preds'][test.s[s_col] == 0].mean()}")
        print(
            f"P( y^=1 | s=0, y=1 ) = {preds_kc['preds'][(test.s[s_col] == 0) & (test.y[y_col] == 1)].mean()}")

        print(f"P( y=1 ) = {test.y[y_col].mean()}")
        print(f"P( y=1 | s=1 ) = {test.y[y_col][test.s[s_col] == 1].mean()}")
        print(
            f"P( y=1 | s=1, y=1 ) = {test.y[y_col][(test.s[s_col] == 1) & (test.y[y_col])].mean()}")
        print(f"P( y=1 | s=0 ) = {test.y[y_col][test.s[s_col] == 0].mean()}")
        print(
            f"P( y=1 | s=0, y=1 ) = {test.y[y_col][(test.s[s_col] == 0) & (test.y[y_col])].mean()}")

        if not flags.strategy == 'flip_y':
            for l, ((_, x_g), (_, s_g), (_, y_g), (_, a_g)) in enumerate(tqdm_notebook(zip(feats_grouped, sens_grouped, labels_grouped, actual_grouped), total=test.x.shape[0], desc="kc ind par")):
                _mod_preds = pd.DataFrame(clf.predict(x_g.drop(["id"], axis=1)), columns=["preds"])
                tt = DataTuple(x=x_g.drop(["id"], axis=1).reset_index(drop=True), s=s_g.drop(["id"], axis=1).reset_index(drop=True), y=a_g.drop(["id"], axis=1).reset_index(drop=True), name=f"Imagined: {train.name}")
                res = metric_per_sensitive_attribute(_mod_preds, tt, ProbPos())
                total += sum(list(diff_per_sensitive_attribute(res).values()))

                if a_g[a_g.columns[0]].values[0] == 1:
                    tpr_total += sum(list(diff_per_sensitive_attribute(res).values()))
                    tpr_count += 1

                acc += 2 - abs(_mod_preds.values - (a_g.drop("id", axis=1).values)).sum()
                _total += (_mod_preds.values[0] ^ _mod_preds.values[1]).sum()

            tqdm.write(f"KC Ind. Parity {_mod.name}: {total/(test.x.shape[0])}")
            tqdm.write(f"KC Ind. Eq Opp {_mod.name}: {tpr_total/tpr_count}")




        s_col = _train.s.columns[0]
        y_col = _train.y.columns[0]

        to_return = DataTuple(x=feats_train.drop(["id"], axis=1), s=s_1_list_train.drop(["id"], axis=1), y=preds_train.drop(["id"], axis=1), name=f"Imagined: {train.name}")

        to_observe = DataTuple(x=feats_train, s=s_1_list_train, y=direct_preds_train, name=f"Imagined: {train.name}")

        _mod = LRCV()
        clf = _mod.fit(train)
        total = 0
        tpr_total = 0
        _total = 0
        acc = 0
        tpr_count = 0
        feats_grouped = feats_test.groupby("id")
        labels_grouped = preds_test.groupby("id")
        sens_grouped = s_1_list_test.groupby("id")
        actual_grouped = actual_labels_test.groupby("id")

        preds_lr = pd.DataFrame(clf.predict(test.x), columns=['preds'])
        tqdm.write(f"lr train acc: {Accuracy().score(preds_lr, test)}")

        res = metric_per_sensitive_attribute(preds_lr, test, ProbPos())
        tqdm.write(f"lr train dp: {res}")
        tqdm.write(f"lr train dp: {diff_per_sensitive_attribute(res).values()}")

        res = metric_per_sensitive_attribute(preds_lr, test, TPR())
        tqdm.write(f"lr train eq op: {res}")
        tqdm.write(f"lr train eq op: {diff_per_sensitive_attribute(res).values()}")

        print(f"P( y^=1 ) = {preds_lr['preds'].mean()}")
        print(f"P( y^=1 | y=1 ) = {preds_lr['preds'][test.y[y_col] == 1].mean()}")
        print(f"P( y^=1 | s=1 ) = {preds_lr['preds'][test.s[s_col] == 1].mean()}")
        print(
            f"P( y^=1 | s=1, y=1 ) = {preds_lr['preds'][(test.s[s_col] == 1) & (test.y[y_col] == 1)].mean()}")
        print(f"P( y^=1 | s=0 ) = {preds_lr['preds'][test.s[s_col] == 0].mean()}")
        print(
            f"P( y^=1 | s=0, y=1 ) = {preds_lr['preds'][(test.s[s_col] == 0) & (test.y[y_col] == 1)].mean()}")

        print(f"P( y=1 ) = {test.y[y_col].mean()}")
        print(f"P( y=1 | s=1 ) = {test.y[y_col][test.s[s_col] == 1].mean()}")
        print(
            f"P( y=1 | s=1, y=1 ) = {test.y[y_col][(test.s[s_col] == 1) & (test.y[y_col])].mean()}")
        print(f"P( y=1 | s=0 ) = {test.y[y_col][test.s[s_col] == 0].mean()}")
        print(
            f"P( y=1 | s=0, y=1 ) = {test.y[y_col][(test.s[s_col] == 0) & (test.y[y_col])].mean()}")

        if not flags.strategy == 'flip_y':
            for l, ((_, x_g), (_, s_g), (_, y_g), (_, a_g)) in enumerate(tqdm_notebook(zip(feats_grouped, sens_grouped, labels_grouped, actual_grouped), total=test.x.shape[0], desc="lr ind par")):
                _mod_preds = pd.DataFrame(clf.predict(x_g.drop(["id"], axis=1)), columns=["preds"])
                tt = DataTuple(x=x_g.drop(["id"], axis=1).reset_index(drop=True), s=s_g.drop(["id"], axis=1).reset_index(drop=True), y=a_g.drop(["id"], axis=1).reset_index(drop=True), name=f"Imagined: {train.name}")
                res = metric_per_sensitive_attribute(_mod_preds, tt, ProbPos())
                total += sum(list(diff_per_sensitive_attribute(res).values()))

                if a_g[a_g.columns[0]].values[0] == 1:
                    tpr_total += sum(list(diff_per_sensitive_attribute(res).values()))
                    tpr_count += 1

                acc += 2 - abs(_mod_preds.values - (a_g.drop("id", axis=1).values)).sum()
                _total += (_mod_preds.values[0] ^ _mod_preds.values[1]).sum()
            tqdm.write(f"LR Ind. Parity {_mod.name}: {total/(test.x.shape[0])}")
            tqdm.write(f"LR Ind. Eq Opp {_mod.name}: {tpr_total/tpr_count}")

        # _lr_lhs = LR()
        # _lr_rhs = LR()
        # lr_clf_lhs = _lr_lhs.fit(train)
        # lr_clf_rhs = _lr_rhs.fit(DataTuple(x=feats_train_encs_og_only, s=train.s, y=train.y))
        # lr_preds_lhs = pd.DataFrame(columns=['preds_lhs'])
        # lr_preds_rhs = pd.DataFrame(columns=['preds_rhs'])
        #
        # _kc_lhs = Kamiran()
        # _kc_rhs = Kamiran()
        # kc_clf_lhs = _kc_lhs.fit(train)
        # kc_clf_rhs = _kc_rhs.fit(DataTuple(x=feats_train_encs_og_only, s=train.s, y=train.y))
        # kc_preds_lhs = pd.DataFrame(columns=['preds_lhs'])
        # kc_preds_rhs = pd.DataFrame(columns=['preds_rhs'])
        #
        # _im_lhs = LR()
        # _im_rhs = LR()
        # im_clf_lhs = _im_lhs.fit(DataTuple(x=feats_train.drop("id", axis=1), s=s_1_list_train.drop('id', axis=1),
        #                                    y=preds_train.drop('id', axis=1)))
        # im_clf_rhs = _im_rhs.fit(DataTuple(x=feats_train_encs_og_only, s=train.s, y=train.y))
        # im_preds_lhs = pd.DataFrame(columns=['preds_lhs'])
        # im_preds_rhs = pd.DataFrame(columns=['preds_rhs'])
        # for _i, _x, _s, _y, _out in tqdm(test_loader, desc="checking"):
        #     _i = _i.to(device)
        #     _x = _x.to(device)
        #     _s = _s.to(device)
        #     _y = _y.to(device)
        #     _out = [out.to(device) for out in _out]
        #
        #     _s_1 = _s.detach().clone()
        #     _s_2 = _s.detach().clone()
        #     feat_enc, feat_dec, feat_s_pred, pred_enc, pred_dec, pred_s_pred, direct_prediction = model(_x, _s_1, _s_2, _s.to(device))
        #     lr_preds_lhs = pd.concat(
        #         [
        #             lr_preds_lhs,
        #             pd.DataFrame(lr_clf_lhs.predict_proba(_x.cpu().numpy())[:,1], columns=["preds_lhs"], dtype=np.float64),
        #         ],
        #         axis='rows',
        #         ignore_index=True,
        #     )
        #     lr_preds_rhs = pd.concat(
        #         [
        #             lr_preds_rhs,
        #             pd.DataFrame(lr_clf_rhs.predict_proba(feat_enc.mean.cpu().numpy())[1],
        #                          columns=['preds_rhs'], dtype=np.int64),
        #         ],
        #         axis='rows',
        #         ignore_index=True,
        #     )
        #     kc_preds_lhs = pd.concat(
        #         [
        #             kc_preds_lhs,
        #             pd.DataFrame(kc_clf_lhs.predict_proba(_x.cpu().numpy())[:,1], columns=["preds_lhs"], dtype=np.float64),
        #         ],
        #         axis='rows',
        #         ignore_index=True,
        #     )
        #     kc_preds_rhs = pd.concat(
        #         [
        #             kc_preds_rhs,
        #             pd.DataFrame(kc_clf_rhs.predict_proba(feat_enc.mean.cpu().numpy())[:,1], columns=['preds_rhs'], dtype=np.float64),
        #         ],
        #         axis='rows',
        #         ignore_index=True,
        #     )
        #     im_preds_lhs = pd.concat(
        #         [
        #             im_preds_lhs,
        #             pd.DataFrame(im_clf_lhs.predict_proba(_x.cpu().numpy())[:,1], columns=["preds_lhs"], dtype=np.float64),
        #         ],
        #         axis='rows',
        #         ignore_index=True,
        #     )
        #     im_preds_rhs = pd.concat(
        #         [
        #             im_preds_rhs,
        #             pd.DataFrame(im_clf_rhs.predict_proba(feat_enc.mean.cpu().numpy())[:,1],
        #                          columns=['preds_rhs'], dtype=np.float64),
        #         ],
        #         axis='rows',
        #         ignore_index=True,
        #     )
        #
        # lr_eval_ind_par = pd.concat([lr_preds_lhs, lr_preds_rhs], axis='columns')
        # kc_eval_ind_par = pd.concat([kc_preds_lhs, kc_preds_rhs], axis='columns')
        # im_eval_ind_par = pd.concat([im_preds_lhs, im_preds_rhs], axis='columns')
        # tqdm.write(f"lr train ind par: {(lr_eval_ind_par['preds_lhs']-lr_eval_ind_par['preds_rhs']).abs().mean()}")
        # tqdm.write(f"lr train ind eq op: {(lr_eval_ind_par[(actual_labels_test[y_col] == 1)]['preds_lhs']-lr_eval_ind_par[(actual_labels_test[y_col] == 1)]['preds_rhs']).abs().mean()}")
        # tqdm.write(f"lr train acc: {Accuracy().score(lr_preds_lhs.applymap(lambda x: 1 if x >= 0.5 else 0), test)}")
        # tqdm.write(f"kc train ind par: {(kc_eval_ind_par['preds_lhs']-kc_eval_ind_par['preds_rhs']).abs().mean()}")
        # tqdm.write(f"kc train ind eq op: {(kc_eval_ind_par[(actual_labels_test[y_col] == 1)]['preds_lhs']-kc_eval_ind_par[(actual_labels_test[y_col] == 1)]['preds_rhs']).abs().mean()}")
        # tqdm.write(f"kc train acc: {Accuracy().score(kc_preds_lhs.applymap(lambda x: 1 if x >= 0.5 else 0), test)}")
        # tqdm.write(f"im train ind par: {(im_eval_ind_par['preds_lhs']-im_eval_ind_par['preds_rhs']).abs().mean()}")
        # tqdm.write(f"im train ind eq op: {(im_eval_ind_par[(actual_labels_test[y_col] == 1)]['preds_lhs']-im_eval_ind_par[(actual_labels_test[y_col] == 1)]['preds_rhs']).abs().mean()}")
        # tqdm.write(f"im train acc: {Accuracy().score(im_preds_lhs.applymap(lambda x: 1 if x >= 0.5 else 0), test)}")





    # from ethicml.visualisation.plot import save_label_plot
    #
    # save_label_plot(to_return, './labels_preds.png')
    # save_label_plot(to_observe, './labels_direct.png')
    # save_label_plot(_train, './labels_og.png')

    if post_process:
        to_return = processor.post(to_return)
        to_observe = processor.post(to_observe)

    return to_return, test#DataTuple(x=feats_train, s=s_1_list_train, y=preds_train, name=f"Imagined: {train.name}")


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
    def __init__(self, in_size: int):
        super().__init__()
        self.hid_1 = nn.Linear(in_size, 100)
        self.bn_1 = nn.BatchNorm1d(100)
        self.hid_2 = nn.Linear(100, 100)
        self.bn_2 = nn.BatchNorm1d(100)
        self.hid_3 = nn.Linear(100, 100)
        self.bn_3 = nn.BatchNorm1d(100)

        self.mu = nn.Linear(100, FEAT_LD)
        self.logvar = nn.Linear(100, FEAT_LD)

    def forward(self, z: torch.Tensor):
        x = self.bn_1(F.relu(self.hid_1(z)))
        # x = F.selu(self.hid_2(x))
        # x = F.selu(self.hid_3(x))
        # if self.training:
        return td.Normal(loc=self.mu(x), scale=F.softplus(self.logvar(x)))
        # else:
        #     return self.mu(x).tanh()


class FeatureDecoder(nn.Module):
    def __init__(self, out_groups):
        super().__init__()
        self.hid_1 = nn.Linear(FEAT_LD + 1, 100)
        self.bn_1 = nn.BatchNorm1d(100)
        self.hid_2 = nn.Linear(100, 100)
        self.bn_2 = nn.BatchNorm1d(100)
        self.hid_3 = nn.Linear(100, 100)
        self.bn_3 = nn.BatchNorm1d(100)

        self.out = nn.ModuleList([nn.Linear(100, len(out)) for out in out_groups])

    def ohe(self, dist: td.OneHotCategorical) -> torch.Tensor:
        am = dist.probs.argmax(1)
        am = am.type(torch.int64).view(-1, 1).cpu()
        one_hots = torch.zeros(dist.probs.shape).scatter_(1, am, 1)
        one_hots = one_hots.view(*am.shape, -1)
        return one_hots.squeeze(1)

    def forward(self, z: td.Distribution, s: torch.Tensor):
        # if self.training:
        #     x = self.bn_1(F.relu(self.hid_1(torch.cat([z.rsample(), s], dim=1))))
        # else:
        x = self.bn_1(F.relu(self.hid_1(torch.cat([z, s], dim=1))))
        # x = F.selu(self.hid_2(x))
        # x = F.selu(self.hid_3(x))
        if self.training:
            return [td.OneHotCategorical(logits=f(x)) for f in self.out]
        else:
            return [self.ohe(td.OneHotCategorical(logits=f(x))) for f in self.out]


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
        # if self.training:
        #     x = self.bn_1(F.relu(self.hid(grad_reverse(z.rsample()))))
        # else:
        x = self.bn_1(F.relu(self.hid(grad_reverse(z))))
        # z = self.bn_1(torch.relu(self.hid_1(z)))
        # z = self.bn_2(torch.relu(self.hid_2(z)))
        x = self.out(x)
        return x


class PredictionEncoder(nn.Module):
    def __init__(self, in_size: int):
        super().__init__()
        self.hid_1 = nn.Linear(in_size, 100)
        self.bn_1 = nn.BatchNorm1d(100)
        self.hid_2 = nn.Linear(100, 100)
        self.bn_2 = nn.BatchNorm1d(100)
        self.hid_3 = nn.Linear(100, 100)
        self.bn_3 = nn.BatchNorm1d(100)
        self.mu = nn.Linear(100, _PRED_LD)
        self.logvar = nn.Linear(100, _PRED_LD)

    def forward(self, z: torch.Tensor):
        x = self.bn_1(F.relu(self.hid_1(z)))
        x = self.bn_2(F.relu(self.hid_2(x)))
        # x = self.bn_3(F.relu(self.hid_3(x)))
        # x = self.mu(x)
        return td.Normal(loc=self.mu(x), scale=F.softplus(self.logvar(x)))


class PredictionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.hid_1 = nn.Linear(_PRED_LD + 1, 20)
        self.bn_1 = nn.BatchNorm1d(20)
        self.hid_2 = nn.Linear(20, 20)
        self.bn_2 = nn.BatchNorm1d(20)
        self.hid_3 = nn.Linear(20, 20)
        self.bn_3 = nn.BatchNorm1d(20)
        self.out = nn.Linear(20, 1)

    def forward(self, z: td.Distribution, s: torch.Tensor):
        x = self.bn_1(F.relu(self.hid_1(torch.cat([z, s], dim=1))))
        x = self.bn_2(F.relu(self.hid_2(x)))
        # x = F.selu(self.hid_3(x))
        # x = z + self.out(x)
        x = self.out(x)#.sigmoid()
        return x#td.Bernoulli(probs=x)


class DirectPredictor(nn.Module):
    def __init__(self, in_size: int):
        super().__init__()
        self.hid = nn.Linear(in_size + 1, 100)
        self.hid_1 = nn.Linear(100, 100)
        self.bn_1 = nn.BatchNorm1d(100)
        self.hid_2 = nn.Linear(100, 100)
        self.bn_2 = nn.BatchNorm1d(100)
        self.bn_3 = nn.BatchNorm1d(100)
        self.out_bn = nn.BatchNorm1d(1)
        self.out = nn.Linear(100, 1)

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        mean = self.bn_1(F.relu(self.hid(torch.cat([x, s], dim=1))))
        mean = self.bn_2(F.relu(self.hid_1(mean)))
        # mean = self.bn_3(F.relu(self.hid_2(mean)))
        mean = self.out(mean)
        return mean #td.Bernoulli(logits=mean)


class PredictionAdv(nn.Module):
    def __init__(self):
        super().__init__()
        self.hid = nn.Linear(_PRED_LD, 100)
        self.hid_1 = nn.Linear(100, 100)
        self.bn_1 = nn.BatchNorm1d(100)
        self.hid_2 = nn.Linear(100, 100)
        self.bn_2 = nn.BatchNorm1d(100)
        self.out = nn.Linear(100, 1)

    def forward(self, z: td.Distribution):
        x = self.bn_1(F.relu(self.hid(grad_reverse(z))))
        # x = self.bn_2(F.relu(self.hid_1(x)))
        # z = self.bn_2(torch.relu(self.hid_2(z)))
        x = self.out(x)
        return x


class Feats(nn.Module):
    def __init__(self, data: CustomDataset):
        super().__init__()

        self.feature_encoder = FeatureEncoder(in_size=data.size)
        self.feature_decoder = FeatureDecoder(data.groups)
        self.feature_adv = FeatureAdv()

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        feat_enc: td.Distribution = self.feature_encoder(x)
        feat_sample = feat_enc.rsample()
        feat_dec = self.feature_decoder(feat_sample, s)
        feat_dec_flip = self.feature_decoder(feat_sample, (1-s))

        feat_s_pred = self.feature_adv(feat_sample)

        return feat_enc, feat_dec, feat_dec_flip, feat_s_pred


class Preds(nn.Module):
    def __init__(self, data: CustomDataset):
        super().__init__()

        self.prediction_encoder = PredictionEncoder(in_size=data.size)
        self.prediction_decoder = PredictionDecoder()

        self.prediction_adv = PredictionAdv()


    def forward(self, x: torch.Tensor, s: torch.Tensor):
        pred_enc: td.Distribution = self.prediction_encoder(x)
        sample = pred_enc.rsample()
        pred_dec: td.Distribution = self.prediction_decoder(sample, s)
        pred_dec_flip: td.Distribution = self.prediction_decoder(sample, (1-s))

        pred_s_pred = self.prediction_adv(sample)

        return pred_enc, pred_dec, pred_dec_flip, pred_s_pred


class Imagine(nn.Module):
    def __init__(self, data: CustomDataset):
        super().__init__()
        self.feats = Feats(data=data)
        self.preds = Preds(data=data)

        self.direct_pred = DirectPredictor(in_size=data.size)


    def forward(self, x, s_1, s_2, _s):
        feat_enc, feat_dec, feat_dec_flip, feat_s_pred = self.feats(x, s_1)
        pred_enc, pred_dec, pred_dec_flip, pred_s_pred = self.preds(x, s_2)

        direct_prediction: td.Distribution = self.direct_pred(x, _s)

        return feat_enc, feat_dec, feat_dec_flip, feat_s_pred, pred_enc, pred_dec, pred_dec_flip, pred_s_pred, direct_prediction


def save_checkpoint(checkpoint, filename, is_best, save_path):
    tqdm.write("===> Saving checkpoint '{}'".format(filename))
    model_filename = save_path / filename
    best_filename = save_path / 'model_best.pth.tar'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    torch.save(checkpoint, model_filename)
    if is_best:
        shutil.copyfile(model_filename, best_filename)
    tqdm.write("===> Saved checkpoint '{}'".format(model_filename))


START_SAVING = 10


def train_model(epoch, model, train_loader, valid_loader, optimizer, device, flags):
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
    best_loss = np.inf

    if epoch > START_SAVING + 1:
        last_epoch = epoch - 1
        filename = 'checkpoint_%03d.pth.tar' % last_epoch
        PATH = Path(".") / "checkpoint" / filename
        dict_ = torch.load(PATH)
        best_loss = dict_['best_loss']

    model.train()
    train_loss = 0
    for batch_idx, (i, data_x, data_s, data_y, out_groups) in enumerate(train_loader):
        data_x = data_x.to(device)
        data_s = data_s.to(device)
        data_s_1 = data_s.to(device)
        data_s_2 = data_s.to(device)
        data_y = data_y.to(device)
        out_groups = [out.to(device) for out in out_groups]

        optimizer.zero_grad()
        feat_enc: td.Distribution
        feat_dec: torch.Tensor
        feat_s_pred: td.Distribution
        pred_enc: td.Distribution
        pred_dec: td.Distribution
        pred_s_pred: td.Distribution
        direct_prediction: td.Distribution
        feat_enc, feat_dec, feat_dec_flip, feat_s_pred, pred_enc, pred_dec, pred_dec_flip, pred_s_pred, direct_prediction = model(
            data_x, data_s_1, data_s_2, data_s.to(device)
        )

        ### Features
        recon_loss = sum([-ohe.log_prob(real) for ohe, real in zip(feat_dec, out_groups)])

        feat_prior = td.Normal(
            loc=torch.zeros(1).to(device), scale=torch.ones(1).to(device)
        )
        feat_kl_loss = td.kl.kl_divergence(feat_enc, feat_prior)

        feat_sens_loss = F.binary_cross_entropy_with_logits(feat_s_pred, data_s)#-feat_s_pred.log_prob(data_s.to(device))
        ###

        ### Predictions
        # if torch.isnan(-direct_prediction.log_prob(data_y).sum()):
        #     print("direct", pred_enc.probs)
        # elif torch.isnan(-pred_dec.log_prob(data_y).sum()):
        #     print("pred", pred_dec.probs)
        pred_loss_1 = F.binary_cross_entropy_with_logits(direct_prediction, data_y)#-direct_prediction.log_prob(data_y)
        pred_loss_2 = F.binary_cross_entropy_with_logits(pred_dec, data_y)#0*-pred_dec.log_prob(data_y)
        pred_loss = (pred_loss_1 + pred_loss_2)#.squeeze()

        pred_prior = \
            td.Normal(loc=torch.zeros(_PRED_LD).to(device), scale=torch.ones(_PRED_LD).to(device))
        #     td.Bernoulli((data_x.new_ones(pred_enc.probs.shape) * 0.5))
        pred_kl_loss = td.kl.kl_divergence(pred_prior, pred_enc).sum(1)
        # pred_kl_loss = kl_b_b(pred_enc.sigmoid(), (data_x.new_ones(pred_enc.shape) * 0.5))

        pred_sens_loss = F.binary_cross_entropy_with_logits(pred_s_pred, data_s)#-pred_s_pred.log_prob(data_s.to(device))
        ###

        ### Direct Pred
        # direct_prediction = td.Bernoulli(probs=torch.ones_like(direct_prediction)*0.5)
        # direct_loss = 0*td.kl.kl_divergence(direct_prediction, pred_dec)
        direct_loss = F.mse_loss(direct_prediction, pred_dec)#torch.zeros_like(direct_prediction.sigmoid())#kl_b_b(direct_prediction.sigmoid(), pred_dec.sigmoid())
        ###

        kl_loss = feat_kl_loss.sum(1) + (pred_kl_loss + direct_loss)#.squeeze(1)
        sens_loss = (feat_sens_loss + pred_sens_loss)

        loss = (recon_loss + kl_loss + sens_loss.squeeze() + pred_loss).mean()

        loss.backward()

        train_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        if batch_idx % 100 == 0:
            tqdm.write(
                f'train Epoch: {epoch} [{batch_idx * len(data_x)}/{len(train_loader.dataset)}'
                f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                f'Loss: {loss.item() / len(data_x):.6f}\t'
                f'recon_loss: {recon_loss.mean().item():.6f}\t'
                f'pred_loss_xs: {pred_loss_1.mean().item():.6f}\t'
                f'pred_loss_ybs: {pred_loss_2.mean().item():.6f}\t'
                f'kld_loss feats: {feat_kl_loss.mean().item():.6f}\t'
                f'kld_loss prior: {pred_kl_loss.mean().item():.6f}\t'
                f'kld_loss outps: {direct_loss.mean().item():.6f}\t'
                f'adv_feat_loss: {feat_sens_loss.mean().item():.6f}\t'
                f'adv_pred_loss: {pred_sens_loss.mean().item():.6f}\t'
            )

    tqdm.write(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

    # model.eval()

    valid_loss = np.inf
    if epoch > START_SAVING:
        valid_loss = 0
        with torch.no_grad():
            for i, data_x, data_s, data_y, out_groups in valid_loader:
                data_x = data_x.to(device)
                data_s_1 = data_s.to(device)
                data_s_2 = data_s.to(device)
                data_y = data_y.to(device)
                out_groups = [out.to(device) for out in out_groups]

                feat_enc, feat_dec, feat_dec_flip, feat_s_pred, pred_enc, pred_dec, pred_dec_flip, pred_s_pred, direct_prediction = model(
                    data_x, data_s_1, data_s_2, data_s.to(device)
                )

                ### Features
                recon_loss = sum([-ohe.log_prob(real) for ohe, real in zip(feat_dec, out_groups)])

                feat_prior = td.Normal(
                    loc=torch.zeros(1).to(device), scale=torch.ones(1).to(device)
                )
                feat_kl_loss = td.kl.kl_divergence(feat_enc, feat_prior)

                feat_sens_loss = F.binary_cross_entropy_with_logits(feat_s_pred,
                                                                    data_s)  # -feat_s_pred.log_prob(data_s.to(device))
                ###

                ### Predictions
                # if torch.isnan(-direct_prediction.log_prob(data_y).sum()):
                #     print("direct", pred_enc.probs)
                # elif torch.isnan(-pred_dec.log_prob(data_y).sum()):
                #     print("pred", pred_dec.probs)
                pred_loss_1 = F.binary_cross_entropy_with_logits(direct_prediction,
                                                                     data_y)  # -direct_prediction.log_prob(data_y)
                pred_loss_2 = F.binary_cross_entropy_with_logits(pred_dec,
                                                                     data_y)  # 0*-pred_dec.log_prob(data_y)
                pred_loss = (pred_loss_1 + pred_loss_2)  # .squeeze()

                pred_prior = \
                    td.Normal(loc=torch.zeros(_PRED_LD).to(device),
                              scale=torch.ones(_PRED_LD).to(device))
                #     td.Bernoulli((data_x.new_ones(pred_enc.probs.shape) * 0.5))
                pred_kl_loss = td.kl.kl_divergence(pred_prior, pred_enc).sum(1)
                # pred_kl_loss = kl_b_b(pred_enc.sigmoid(), (data_x.new_ones(pred_enc.shape) * 0.5))

                pred_sens_loss = F.binary_cross_entropy_with_logits(pred_s_pred,
                                                                    data_s)  # -pred_s_pred.log_prob(data_s.to(device))
                ###

                ### Direct Pred
                # direct_prediction = td.Bernoulli(probs=torch.ones_like(direct_prediction)*0.5)
                # direct_loss = 0*td.kl.kl_divergence(direct_prediction, pred_dec)
                direct_loss = F.mse_loss(direct_prediction, pred_dec)#(-F.binary_cross_entropy_with_logits(direct_prediction,
                                                                       # data_y) + F.binary_cross_entropy_with_logits(
                    # pred_dec,
                    # data_y))  # torch.zeros_like(direct_prediction.sigmoid())#kl_b_b(direct_prediction.sigmoid(), pred_dec.sigmoid())
                ###

                kl_loss = feat_kl_loss.sum(1) + (pred_kl_loss + direct_loss)#.squeeze(1)
                sens_loss = (feat_sens_loss + pred_sens_loss)

                valid_loss = (recon_loss + kl_loss + sens_loss.squeeze() + pred_loss).mean()

    tqdm.write(f"Validation loss: {valid_loss} \t Best Loss: {best_loss} \t {(valid_loss/best_loss)*100}")
    is_best = valid_loss < best_loss
    best_loss = min(valid_loss, best_loss)
    if is_best:
        tqdm.write(f"Best saved at epoch {epoch}")

    # Save checkpoint
    save_path = Path(".") / "checkpoint"
    model_filename = 'checkpoint_%03d.pth.tar' % epoch
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_loss': best_loss,
    }
    save_checkpoint(checkpoint, model_filename, is_best, save_path)


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
    parser.add_argument("--sample", type=int, required=True)
    parser.add_argument("--start_from", type=int, required=True)
    parser.add_argument("--strategy", type=str, required=True)
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
        sample=args.sample,
        start_from=args.start_from,
        strategy=args.strategy,
    )
    save_transformations(train_and_transform(train, test, flags), args)


def kl_b_b(p, q):
    t1 = p * (p / q).log()
    t1[q == 0] = np.inf
    t1[p == 0] = 0
    t2 = (1 - p) * ((1 - p) / (1 - q)).log()
    t2[q == 1] = np.inf
    t2[p == 1] = 0
    return t1 + t2

if __name__ == "__main__":
    main()
