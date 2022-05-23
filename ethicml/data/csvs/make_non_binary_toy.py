"""Script to generate the toy dataset."""
from pathlib import Path

import numpy as np
import pandas as pd


def main(seed: int, samples: int) -> None:
    """Make a toy dataset that can be used to check an algorithm runs."""
    rng = np.random.default_rng(seed)

    feat1_g1: np.ndarray = rng.standard_normal(size=samples // 2) - 0.1
    feat1_g2: np.ndarray = rng.standard_normal(size=samples // 2) + 0.5

    feat2_g1: np.ndarray = rng.standard_normal(samples // 2) * 0.9
    feat2_g2: np.ndarray = rng.standard_normal(samples // 2) * 1.1

    disc_1 = rng.choice(a=['a', 'b', 'c', 'd', 'e'], size=samples, replace=True)
    disc_2 = rng.choice(a=['x', 'y', 'z'], size=samples, replace=True)

    sensitive_attr_g1 = np.full(feat1_g1.shape, 0)
    sensitive_attr_g2 = np.full(feat1_g2.shape, 1)
    feat1 = np.concatenate((feat1_g1, feat1_g2))
    feat2 = np.concatenate((feat2_g1, feat2_g2))
    sensitive_attr = np.concatenate((sensitive_attr_g1, sensitive_attr_g2))

    # There has to be a better way of dong this.
    decision = rng.multinomial(
        n=1,
        pvals=rng.dirichlet(
            alpha=(
                abs(feat1).mean(),
                abs(feat2).mean(),
                abs(feat1).mean() + abs(feat2).mean(),
                abs(feat1_g1).mean(),
                abs(feat2_g2).mean(),
            ),
            size=samples,
        ).mean(axis=0),
        size=samples,
    ).argmax(axis=1)

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
    cols_to_dum = ["sensitive-attr", "disc_1", "disc_2"]
    df = pd.get_dummies(df, columns=cols_to_dum)

    # Shuffle the data,
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Save the CSV
    df.to_csv(str(Path(__file__).parent / "nbt.csv"), index=False)


if __name__ == "__main__":
    SEED = 888
    SAMPLES = 400
    main(SEED, SAMPLES)
