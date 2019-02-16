from com.sbk.digit.nn.nn_digit_recogniton import NNDigitRecognition
import numpy as np
import mnist


num_classes = 10
train_images = mnist.train_images() #[60000, 28, 28]
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

X_train = train_images.reshape(train_images.shape[0], train_images.shape[1]*train_images.shape[2]).astype('float32')
x_train = X_train / 255
y_train = np.eye(num_classes)[train_labels]

X_test = test_images.reshape(test_images.shape[0], test_images.shape[1]*test_images.shape[2]).astype('float32')
x_test = X_test / 255
y_test = test_labels


n = NNDigitRecognition()
n.train(x_train, y_train)

print("Testing...")
n.test(x_test, y_test)

print("Digit from me:")
print(n.recognize_digit("images/test_image_0.png"))
print(n.recognize_digit("images/test_image_1.png"))
print(n.recognize_digit("images/test_image_2.png"))
print(n.recognize_digit("images/test_image_3.png"))
print(n.recognize_digit("images/test_image_4.png"))
print(n.recognize_digit("images/test_image_5.png"))
print(n.recognize_digit("images/test_image_6.png"))
print(n.recognize_digit("images/test_image_7.png"))
print(n.recognize_digit("images/test_image_8.png"))
print(n.recognize_digit("images/test_image_9.png"))
print("Digit from me(another party):")
print(n.recognize_digit("images/test_image_0_1.png"))
print(n.recognize_digit("images/test_image_1_1.png"))
print(n.recognize_digit("images/test_image_2_1.png"))
print(n.recognize_digit("images/test_image_3_1.png"))
print(n.recognize_digit("images/test_image_4_1.png"))
print(n.recognize_digit("images/test_image_5_1.png"))
print(n.recognize_digit("images/test_image_6_1.png"))