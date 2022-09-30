import math
import numpy as np


def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


def applyFilter(points, t, min_cutoff, beta, skipPoints = []):
    filtered = np.empty_like(points)
    filtered[0] = points[0]
    one_euro_filter = OneEuroFilter(t[0], points[0], min_cutoff=min_cutoff, beta=beta)
    
    for i in range(1, points.shape[0]):
        filtered[i] = one_euro_filter(t[i], points[i])
        
    for i in range(1, points.shape[0]):
        for skipPoint in skipPoints:
            filtered[i, skipPoint] = points[i, skipPoint]

    return filtered


def applyFilterOnMesh(landmarks):
    shape_1, shape_2, shape_3 = landmarks.shape
    xs = landmarks[:,:,0].reshape((shape_1, shape_2))
    ys = landmarks[:,:,1].reshape((shape_1, shape_2))
    zs = landmarks[:,:,2].reshape((shape_1, shape_2))
    t = np.linspace(0, xs.shape[0] / 25, xs.shape[0])
    xs_hat = applyFilter(xs, t, 0.005, 0.7)
    ys_hat = applyFilter(ys, t, 0.005, 0.7, mouthPoints + chins)
    ys_hat = applyFilter(ys_hat, t, 0.000001, 1.5, rest)
    zs_hat = applyFilter(zs, t, 0.005, 0.7)
    combine = np.stack(((xs_hat, ys_hat, zs_hat)), axis=2)

    return combine


class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = x0
        self.dx_prev = dx0
        self.t_prev = float(t0)

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat
