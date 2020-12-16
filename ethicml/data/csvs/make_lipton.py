"""Generate Lipton's Synthetic Data."""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 8
SAMPLES = 2_000


def main():
    """Generate the data as specified in section 4.1 of https://arxiv.org/pdf/1711.07076.pdf ."""
    rng = np.random.default_rng(SEED)

    z = rng.binomial(1, 0.5, SAMPLES)

    hair_length_0 = 35 * rng.beta(2, 7, SAMPLES)
    hair_length_1 = 35 * rng.beta(2, 2, SAMPLES)
    hair_length = np.where(z == 1, hair_length_1, hair_length_0)

    pois = rng.poisson((25 + 6 * z))
    norm = rng.normal(20, 0.2, SAMPLES)
    work_experience = pois - norm

    p = 1 / (1 + (np.exp(-(-25.5 + 2.5 * work_experience))))

    y = 2 * rng.binomial(1, p) - 1

    # hair_length = pd.Series(hair_length, name="hair_length", index=range(len(hair_length)))
    # work_experience = pd.Series(
    #     work_experience, name="work_experience", index=range(len(work_experience))
    # )
    # z = pd.Series(z, name="sens", index=range(len(z)))
    # y = pd.Series(y, name="outcome", index=range(len(y)))

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
