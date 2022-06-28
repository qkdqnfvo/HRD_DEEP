import numpy as np

class Activation:
    def __init__(self):
        pass
    
    def softmax(self, x):
        if x.ndim == 1:
            x = x - max(x)
            return np.exp(x) / np.sum(np.exp(x))
        else:
            x = (x.T - np.max(x, axis=1)).T
            x = (np.exp(x).T / np.sum(np.exp(x), axis=1)).T
            return x

    def sigmoid(self, x):
        return 1 / 1 + np.exp(-x)

    def relu(self, x):
        return np.maximum(0, x) # 0과 x 중 큰 값을 리턴(각 요소마다)
        # return np.where(x > 0, x, 0)

    def linear(self, x):
        return x