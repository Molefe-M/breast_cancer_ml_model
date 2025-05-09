# tests/integration/test_scoringflow.py
import unittest
from src.score import ModelScoring
import pandas as pd

class TestScoringFlow(unittest.TestCase):
    def test_scoring_flow(self):
        input_df = pd.read_csv("tests\\test_data\\test_input_dataframe.csv", index_col=False)
        expected_df = pd.read_csv("tests\\test_data\\expected_model_output.csv", index_col=False)
        predicted_df = ModelScoring(input_df).execute()
        pd.testing.assert_frame_equal(predicted_df, expected_df)

if __name__ == "__main__":
    unittest.main()