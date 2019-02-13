import unittest

from com.sbk.digit.own_digit_recognition import OwnDigitRecognition


class TestOwnDigitRecognition(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dr = OwnDigitRecognition()
        dr.train_neural_net()
        cls.net = dr