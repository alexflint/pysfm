import numpy as np

# Within this window, the Cauchy function is approximated as linear
kLinearWindowAboutZero = 1e-5

class CauchyRobustifier(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def evaluate(self, x):
        assert np.isscalar(x)
        s = self.sigma
        return np.log(1. + x/(s*s))

    def cost(self, x):
        assert np.shape(x) == (2,)
        x = np.asarray(x)
        r = np.linalg.norm(x)
        if r < kLinearWindowAboutZero:
            return x / self.sigma

        s = self.sigma
        e = np.sqrt(np.log(1. + r*r / (s*s)))
        return x * e / r

    def Jcost(self, x):
        assert np.shape(x) == (2,)
        x = np.asarray(x)
        r = np.linalg.norm(x)
        if r < kLinearWindowAboutZero:
            return np.eye(2) / self.sigma

        xx = np.outer(x,x)
        s = self.sigma
        e = np.sqrt(np.log(1. + r*r / (s*s)))
        I = np.eye(2)
        return xx / (r*e*(r*r + s*s)) + (r*I - xx/r) * e/(r*r)


class NonRobustifier(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def evaluate(self, x):
        return x

    def cost(self, x):
        assert np.shape(x) == (2,)
        return x

    def Jcost(self, x):
        assert np.shape(x) == (2,)
        return np.eye(2)
