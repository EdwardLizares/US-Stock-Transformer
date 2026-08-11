import unittest

from data_processor import process_data

class TestDataProcessor1(unittest.TestCase):
    def test_process_data(self):
        # Test that the process_data function returns a DataFrame
        df = process_data("test_data/case_1.parquet")
        self.assertIsInstance(df, pd.DataFrame)

if __name__ == "__main__":
    unittest.main()
