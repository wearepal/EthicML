"""Transparently show how the UCI Adult dataset was generated from the raw download."""

import numpy as np
import pandas as pd


def run_generate_crime() -> None:
    """Generate the UCI Communities and Crime dataset from scratch."""
    # We use data from the UCI repository directly and show how the final csv is generated.
    # To use the .arff file from K&C 2012, first download the .arff file from
    # https://sites.google.com/site/faisalkamiran/ which is referenced in their 2012 paper
    # available here: https://core.ac.uk/download/pdf/81728147.pdf
    # then, uncomment the following code.
    #
    # from scipy.io import arff
    # import pandas as pd
    # data = arff.loadarff('./raw/communities.arff')
    # data = pd.DataFrame(data[0])
    # for col in data.columns:
    #     if data[col].dtype == np.object:
    #         data[col] = data[col].astype(int)
    # data = data.replace(r"^\s*\?+\s*$", np.nan, regex=True).dropna()
    # data = data.sample(frac=1.0, random_state=888).reset_index(drop=True)
    # data.to_csv("./crime.csv", index=False)

    # Load the data
    data = pd.read_csv("raw/communities.data")

    columns = [
        "state",
        "county",
        "community",
        "communityname",
        "fold",
        "population",
        "householdsize",
        "racepctblack",
        "racePctWhite",
        "racePctAsian",
        "racePctHisp",
        "agePct12t21",
        "agePct12t29",
        "agePct16t24",
        "agePct65up",
        "numbUrban",
        "pctUrban",
        "medIncome",
        "pctWWage",
        "pctWFarmSelf",
        "pctWInvInc",
        "pctWSocSec",
        "pctWPubAsst",
        "pctWRetire",
        "medFamInc",
        "perCapInc",
        "whitePerCap",
        "blackPerCap",
        "indianPerCap",
        "AsianPerCap",
        "OtherPerCap",
        "HispPerCap",
        "NumUnderPov",
        "PctPopUnderPov",
        "PctLess9thGrade",
        "PctNotHSGrad",
        "PctBSorMore",
        "PctUnemployed",
        "PctEmploy",
        "PctEmplManu",
        "PctEmplProfServ",
        "PctOccupManu",
        "PctOccupMgmtProf",
        "MalePctDivorce",
        "MalePctNevMarr",
        "FemalePctDiv",
        "TotalPctDiv",
        "PersPerFam",
        "PctFam2Par",
        "PctKids2Par",
        "PctYoungKids2Par",
        "PctTeen2Par",
        "PctWorkMomYoungKids",
        "PctWorkMom",
        "NumIlleg",
        "PctIlleg",
        "NumImmig",
        "PctImmigRecent",
        "PctImmigRec5",
        "PctImmigRec8",
        "PctImmigRec10",
        "PctRecentImmig",
        "PctRecImmig5",
        "PctRecImmig8",
        "PctRecImmig10",
        "PctSpeakEnglOnly",
        "PctNotSpeakEnglWell",
        "PctLargHouseFam",
        "PctLargHouseOccup",
        "PersPerOccupHous",
        "PersPerOwnOccHous",
        "PersPerRentOccHous",
        "PctPersOwnOccup",
        "PctPersDenseHous",
        "PctHousLess3BR",
        "MedNumBR",
        "HousVacant",
        "PctHousOccup",
        "PctHousOwnOcc",
        "PctVacantBoarded",
        "PctVacMore6Mos",
        "MedYrHousBuilt",
        "PctHousNoPhone",
        "PctWOFullPlumb",
        "OwnOccLowQuart",
        "OwnOccMedVal",
        "OwnOccHiQuart",
        "RentLowQ",
        "RentMedian",
        "RentHighQ",
        "MedRent",
        "MedRentPctHousInc",
        "MedOwnCostPctInc",
        "MedOwnCostPctIncNoMtg",
        "NumInShelters",
        "NumStreet",
        "PctForeignBorn",
        "PctBornSameState",
        "PctSameHouse85",
        "PctSameCity85",
        "PctSameState85",
        "LemasSwornFT",
        "LemasSwFTPerPop",
        "LemasSwFTFieldOps",
        "LemasSwFTFieldPerPop",
        "LemasTotalReq",
        "LemasTotReqPerPop",
        "PolicReqPerOffic",
        "PolicPerPop",
        "RacialMatchCommPol",
        "PctPolicWhite",
        "PctPolicBlack",
        "PctPolicHisp",
        "PctPolicAsian",
        "PctPolicMinor",
        "OfficAssgnDrugUnits",
        "NumKindsDrugsSeiz",
        "PolicAveOTWorked",
        "LandArea",
        "PopDens",
        "PctUsePubTrans",
        "PolicCars",
        "PolicOperBudg",
        "LemasPctPolicOnPatr",
        "LemasGangUnitDeploy",
        "LemasPctOfficDrugUn",
        "PolicBudgPerPop",
        "ViolentCrimesPerPop",
    ]

    data.columns = pd.Index(columns)

    for col in data.columns:
        if data[col].dtype == np.object:  # type: ignore[attr-defined]
            data[col] = data[col].str.strip()

    # Drop NaNs
    data = data.replace(r"^\s*\?+\s*$", np.nan, regex=True).dropna(axis=1)

    # OHE
    data = pd.get_dummies(data, columns=["state"])

    # Add a new column, ">0.06black" as in:
    # Faisal Kamiran and Toon Calders. 2012. Data preprocessing techniques for classification without discrimination.
    # Knowledge and Information Systems 33, 1 (2012), 1–33.
    data[">0.06black"] = data["racepctblack"] > 0.06
    data[">0.06black"] = data[">0.06black"].astype(int)

    # Add a 'class' label of high violent crime as in K&C 2012
    data["high_crime"] = data["ViolentCrimesPerPop"] > 0.25
    data["high_crime"] = data["high_crime"].astype(int)

    # Shuffle the data
    data = data.sample(frac=1.0, random_state=888).reset_index(drop=True)

    # Save the CSV
    data.to_csv("./crime.csv", index=False)

    print(list(data.columns))


if __name__ == "__main__":
    run_generate_crime()
