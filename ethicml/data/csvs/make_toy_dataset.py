"""Script to generate the toy dataset."""
from pathlib import Path
import random

import numpy as np
import pandas as pd


def main(seed: int, samples: int) -> None:
    """Make a toy dataset that can be used to check an algorithm runs."""
    np.random.seed(seed)
    random.seed(seed)

    feat1_g1: np.ndarray = np.random.randn(samples // 2) - 0.1
    feat1_g2: np.ndarray = np.random.randn(samples // 2) + 0.5

    feat2_g1: np.ndarray = np.random.randn(samples // 2) * 0.9
    feat2_g2: np.ndarray = np.random.randn(samples // 2) * 1.1

    disc_1 = np.random.choice(['a', 'b', 'c', 'd', 'e'], [samples], replace=True)
    disc_2 = np.random.choice(['x', 'y', 'z'], [samples], replace=True)

    sensitive_attr_g1 = np.full(feat1_g1.shape, 0)
    sensitive_attr_g2 = np.full(feat1_g2.shape, 1)
    feat1 = np.concatenate((feat1_g1, feat1_g2))
    feat2 = np.concatenate((feat2_g1, feat2_g2))
    sensitive_attr = np.concatenate((sensitive_attr_g1, sensitive_attr_g2))

    # There has to be a better way of dong this.
    decision = np.array([1 if v > 0 else 0 for v in feat1 + feat2 > 0])
    df = pd.DataFrame(
        data={
            "decision": decision,
            "sensitive-attr": sensitive_attr,
            "a1": feat1,
            "a2": feat2,
            "disc_1": disc_1,
            "disc_2": disc_2,
        }
    )
    df = pd.get_dummies(df)

    # Shuffle the data,
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Save the CSV
    df.to_csv(str(Path(__file__).parent / "toy.csv"), index=False)


if __name__ == "__main__":
    SEED = 888
    SAMPLES = 400
    main(SEED, SAMPLES)
