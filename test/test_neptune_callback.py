import unittest
from unittest.mock import MagicMock
from flexcv.fold_logging import CustomNeptuneCallback
from flexcv.run import Run

class TestCustomNeptuneCallback(unittest.TestCase):
    def setUp(self):
        self.callback = CustomNeptuneCallback(run=Run())

    def test_init(self):
        self.assertIsInstance(self.callback, CustomNeptuneCallback)

if __name__ == "__main__":
    unittest.main()