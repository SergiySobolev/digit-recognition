from scipy.io import loadmat


class ScratchDigitRecognition:

    def train_neural_net(self):
        data = loadmat('datasets/ex4data1.mat')
        X = data['X']
        y = data['y']