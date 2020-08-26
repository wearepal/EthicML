"""Generate Synthetic data."""
from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    """As there is no np.sigmoid."""
    return 1 / (1 + np.exp(-x))


def main() -> None:
    r"""Make synthetic data.

    Generate synthetic data that conforms to 'Scenario 1'

    Scenario 1
    ----------

    +-----+   +-----+
    | Y_1 +<--+ X_1 +-+
    +-----+   +-----+ |   +-----+   +---+
                      +-->+ Y_3 |   | S |
    +-----+   +-----+ |   +-----+   +---+
    | Y_2 +<--+ X_2 +-+
    +-----+   +-----+

    In this scenario, there are two input variables, X_1 and X_2.
    There are three outcome variables, Y_1, Y_2 & Y_3.
    There is one sensitive attribute, S, which is independent of all X & Y.

    We have:
    X_1 ~ N(3, 2)
    X_2 ~ N(-1.5, 4)

    Y_1 ~ B(sigmoid(X_1)))
    Y_2 ~ B(sigmoid(X_2))
    Y_3 ~ B(sigmoid((X_1 + X_2)/2))

    S ~ B(0.5)

    """
    seed = 0
    samples = 1_000

    np.random.seed(seed)

    x_1 = np.random.normal(3, 2, samples)
    x_2 = np.random.normal(-1.5, 4, samples)

    y_1 = np.random.binomial(1, sigmoid(x_1))
    y_2 = np.random.binomial(1, sigmoid(x_2))
    y_3 = np.random.binomial(1, sigmoid(x_1 + x_2))

    s = np.random.binomial(1, 0.5, samples)

    df = pd.DataFrame(data={"x1": x_1, "x2": x_2, "s": s, "y1": y_1, "y2": y_2, "y3": y_3})

    # Shuffle the data,
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Save the CSV
    df.to_csv(str(Path(__file__).parent / "synthetic_scenario_1.csv"), index=False)


if __name__ == "__main__":
    main()
