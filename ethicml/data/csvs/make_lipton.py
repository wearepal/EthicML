"""Generate Lipton's Synthetic Data."""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 8
SAMPLES = 2_000


def main() -> None:
    """Generate the data as specified in section 4.1 of https://arxiv.org/pdf/1711.07076.pdf ."""
    rng = np.random.default_rng(SEED)

    z = rng.binomial(1, 0.5, SAMPLES)

    hair_length_0: np.ndarray = 35 * rng.beta(2, 7, SAMPLES)
    hair_length_1: np.ndarray = 35 * rng.beta(2, 2, SAMPLES)
    hair_length = np.where(z == 1, hair_length_1, hair_length_0)

    pois = rng.poisson(25 + 6 * z)
    norm = rng.normal(20, 0.2, SAMPLES)
    work_experience: np.ndarray = pois - norm

    p = 1 / (1 + (np.exp(-(-25.5 + 2.5 * work_experience))))

    y = 2 * rng.binomial(1, p) - 1

    df = pd.DataFrame(
        data={
            "hair_length": hair_length,
            "work_experience": work_experience,
            "sens": z,
            "hired": y,
        }
    )

    df.to_csv(str(Path(__file__).parent / "lipton.csv"), index=False)


if __name__ == '__main__':
    main()
