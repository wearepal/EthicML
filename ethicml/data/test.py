"""
Class to describe features of the Test dataset
"""

from ethicml.data.dataset import Dataset


class Test(Dataset):
    def get_filename(self):
        return "test.csv"

    def get_feature_split(self):
        return {
            "x": ["a1", "a2"],
            "s": ["s"],
            "y": ["y"]
        }
