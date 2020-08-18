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

             +----------------+
             |                |
             |                v
    +-----+  |   +-----+    +---+
    | X_1 |--+-->| X_2 | -->| Y |
    +-----+      +-----+    +---+

                            +---+
                            | S |
                            +---+

    In this scenario, there are two input variables, X_1 and X_2.
    There is one outcome variable, Y.
    There is one sensitive attribute, S, which is independent of X_1, X_2 & Y.

    We have:
    X_1 ~ N(0, 1)
    X_2 ~ N(-1.5, 4) + X_1
    Y ~ B(sigmoid((X_1 + X_2)/2))
    S ~ B(0.5)

    """
    seed = 0
    samples = 1_000

    np.random.seed(seed)

    x_1 = np.random.normal(0, 1, samples)
    x_2 = np.random.normal(-1.5, 4, samples) + x_1

    p = sigmoid((x_1 + x_2) / 2)
    y = np.random.binomial(1, p)

    s = np.random.binomial(1, 0.5, samples)

    print(s.mean(), y.mean())

    df = pd.DataFrame(data={"x1": x_1, "x2": x_2, "s": s, "y": y,})

    # Shuffle the data,
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Save the CSV
    df.to_csv(str(Path(__file__).parent / "synthetic_scenario_1.csv"), index=False)


if __name__ == "__main__":
    main()
