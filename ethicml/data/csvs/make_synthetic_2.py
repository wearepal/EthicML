"""Generate Synthetic data."""
from pathlib import Path

import numpy as np
import pandas as pd

from ethicml.data.csvs.utility import sigmoid


def main() -> None:
    r"""Make synthetic data.

    Generate synthetic data that conforms to 'Scenario 2'

    Scenario 2
    ----------

               +---+
               | S |
               +-+-+
                 |
                 v
    +-----+   +--+--+
    | Y_1 +<--+ X_1 +-+
    +-----+   +-----+ |   +-----+
                      +-->+ Y_3 |
    +-----+   +-----+ |   +-----+
    | Y_2 +<--+ X_2 +-+
    +-----+   +-----+

    In this scenario, there are two input variables, X_1 and X_2.
    There are three outcome variables, Y_1, Y_2 & Y_3.
    There is one sensitive attribute, S, which is independent of X_2 & Y_2, but not independent of
    X_1, Y_1, or Y_3.

    We have:
    S ~ B(0.5)

    X_1 ~ N(0, 0.5) + 2*S
    X_2 ~ N(-1.5, 4)

    Y_1 ~ B(sigmoid(X_1)))
    Y_2 ~ B(sigmoid(X_2))
    Y_3 ~ B(sigmoid((X_1 + X_2)/2))

    """
    seed = 0
    samples = 100_000

    np.random.seed(seed)

    s = np.random.binomial(1, 0.6, samples)

    x_1f = np.random.normal(0, 0.5, samples)
    x_1: np.ndarray = x_1f + (s * 2 - 1)
    x_2 = np.random.normal(-1, 3, samples)
    x_2f = x_2

    y_1 = np.random.binomial(1, sigmoid(x_1))
    y_1f = np.random.binomial(1, sigmoid(x_1f))
    y_2 = np.random.binomial(1, sigmoid(x_2))
    y_2f = y_2
    p: np.ndarray = x_1 + x_2
    pf: np.ndarray = x_1f + x_2f
    y_3 = np.random.binomial(1, sigmoid(p))
    y_3f = np.random.binomial(1, sigmoid(pf))

    noise_1 = np.random.normal(0, 4, samples)
    noise_2 = np.random.normal(3, 7, samples)

    print(
        s.mean(),
        x_1.mean(),
        x_1f.mean(),
        x_2.mean(),
        x_2f.mean(),
        y_1.mean(),
        y_1f.mean(),
        y_2.mean(),
        y_2f.mean(),
        y_3.mean(),
        y_3f.mean(),
    )

    df = pd.DataFrame(
        data={
            "x1": x_1,
            "x1f": x_1f,
            "x2": x_2,
            "x2f": x_2f,
            "s": s,
            "y1": y_1,
            "y1f": y_1f,
            "y2": y_2,
            "y2f": y_2f,
            "y3": y_3,
            "y3f": y_3f,
            "n1": noise_1,
            "n2": noise_2,
        }
    )

    # Shuffle the data,
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Save the CSV
    df.to_csv(str(Path(__file__).parent / "synthetic_scenario_2.csv"), index=False)


if __name__ == "__main__":
    main()
