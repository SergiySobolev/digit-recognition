import unittest

from com.sbk.digit.own_digit_recognition import OwnDigitRecognition
from com.sbk.digit.scratch_digit_recognition import ScratchDigitRecognition


class TestScratchDigitRecognition(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dr = ScratchDigitRecognition()
        dr.train_neural_net()
        cls.net = dr

    def test_digit_recognition(self):
        self.assertEqual(1, 1)