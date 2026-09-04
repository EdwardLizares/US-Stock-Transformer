import unittest
import pandas as pd

from archived_scripts.app import App

class TestApp(unittest.TestCase):
    def test_process_data(self):
        app = App()

if __name__ == "__main__":
    unittest.main()
