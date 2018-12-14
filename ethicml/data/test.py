"""
Class to describe features of the Test dataset
"""
from typing import List, Dict
from .dataset import Dataset


class Test(Dataset):

    def get_dataset_name(self) -> str:
        return "Test"

    def get_filename(self) -> str:
        return "test.csv"

    def get_feature_split(self) -> Dict[str, List[str]]:
        return {
            "x": ["a1", "a2"],
            "s": ["s"],
            "y": ["y"]
        }
