import unittest

from parameterized import parameterized

from com.sbk.digit.nn.nn_digit_recogniton import NNDigitRecognition


class TestScratchDigitRecognition(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dr = NNDigitRecognition()
        dr.train_on_default_data()
        cls.net = dr

    @parameterized.expand([
        ["Test 0", 'images/test_image_0.png', 0],
        ["Test 1", 'images/test_image_1.png', 1],
        ["Test 2", 'images/test_image_2.png', 2],
        ["Test 3", 'images/test_image_3.png', 3],
        ["Test 4", 'images/test_image_4.png', 4],
        ["Test 5", 'images/test_image_5.png', 5],
        ["Test 6", 'images/test_image_6.png', 6],
        ["Test 8", 'images/test_image_8.png', 8],
        ["Test 9", 'images/test_image_9.png', 9],
        ["Test 1-1", 'images/test_image_1_1.png', 1],
        ["Test 2-1", 'images/test_image_2_1.png', 2],
        ["Test 3-1", 'images/test_image_3_1.png', 3],
        ["Test 4-1", 'images/test_image_4_1.png', 4],
        ["Test 6-1", 'images/test_image_6_1.png', 6]
    ])
    def test_digit_recognition(self, name, img, expected_label):
        self.assertEqual(expected_label, self.net.recognize_digit(img))

    @parameterized.expand([
        ["Test 0", 'images/test_image_0.png', 0],
        ["Test 1", 'images/test_image_1.png', 1],
        ["Test 2", 'images/test_image_2.png', 2],
        ["Test 3", 'images/test_image_3.png', 3],
        ["Test 4", 'images/test_image_4.png', 4],
        ["Test 5", 'images/test_image_5.png', 5],
        ["Test 6", 'images/test_image_6.png', 6],
        ["Test 7", 'images/test_image_7.png', 7],
        ["Test 8", 'images/test_image_8.png', 8],
        ["Test 9", 'images/test_image_9.png', 9],
        ["Test 0-1", 'images/test_image_0_1.png', 0],
        ["Test 1-1", 'images/test_image_1_1.png', 1],
        ["Test 2-1", 'images/test_image_2_1.png', 2],
        ["Test 3-1", 'images/test_image_3_1.png', 3],
        ["Test 4-1", 'images/test_image_4_1.png', 4],
        ["Test 5-1", 'images/test_image_5_1.png', 5],
        ["Test 6-1", 'images/test_image_6_1.png', 6]
    ])
    def test_digit_recognition_pobability(self, name, img, expected_label):
        self.assertTrue(expected_label in self.net.get_most_probable_predictions_for_digit(img))