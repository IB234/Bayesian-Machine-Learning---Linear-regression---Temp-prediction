import matplotlib
# needed to solve problem with matplotlib using pycharm.
matplotlib.use('TkAgg')
import numpy as np
from matplotlib import pyplot as plt
from typing import Callable


def polynomial_basis_functions(degree: int) -> Callable:
    def pbf(x: np.ndarray):
        x = np.asarray(x).reshape(-1)
        N = x.shape[0]
        d = degree
        scale = d if d > 0 else 1.0
        cols = [np.ones(N)]
        if d >= 1:
            t_scaled = x / scale
            for i in range(1, d + 1):
                cols.append(t_scaled ** i)
        H = np.stack(cols, axis=1)
        return H

    return pbf


def fourier_basis_functions(num_freqs: int) -> Callable:
    def fbf(x: np.ndarray):
        x = np.asarray(x).reshape(-1)
        N = x.shape[0]
        K = int(num_freqs)
        cols = [np.ones(N)]
        freq_factor = 2 * np.pi / 24.0
        for k in range(1, K + 1):
            cols.append(np.cos(freq_factor * k * x))
        for k in range(1, K + 1):
            cols.append(np.sin(freq_factor * k * x))
        H = np.stack(cols, axis=1)
        return H

    return fbf


def spline_basis_functions(knots: np.ndarray) -> Callable:
    knots = np.asarray(knots).reshape(-1)

    def csbf(x: np.ndarray):
        x = np.asarray(x).reshape(-1)
        N = x.shape[0]
        cols = [np.ones(N), x, x ** 2, x ** 3]
        for xi in knots:
            diff = x - xi
            pos = np.maximum(diff, 0.0)
            cols.append(pos ** 3)
        H = np.stack(cols, axis=1)
        return H

    return csbf


def learn_prior(hours: np.ndarray, temps: np.ndarray, basis_func: Callable) -> tuple:
    thetas = []
    for i, t in enumerate(temps):
        ln = LinearRegression(basis_func).fit(hours, t)
        thetas.append(ln.theta.copy())
    thetas = np.array(thetas)
    mu = np.mean(thetas, axis=0)
    cov = (thetas - mu[None, :]).T @ (thetas - mu[None, :]) / thetas.shape[0]
    cov += 1e-8 * np.eye(cov.shape[0])
    return mu, cov


class BayesianLinearRegression:
    def __init__(self, theta_mean: np.ndarray, theta_cov: np.ndarray, sig: float, basis_functions: Callable):
        self.theta_mean = np.asarray(theta_mean).reshape(-1)
        self.theta_cov = np.asarray(theta_cov).astype(np.float64)
        self.sig = float(sig)
        self.basis = basis_functions
        self.theta_mean_post = None
        self.theta_cov_post = None
        self.dim = self.theta_mean.shape[0]

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianLinearRegression':
        X = np.asarray(X).reshape(-1)
        y = np.asarray(y).reshape(-1)
        H = self.basis(X)
        try:
            S0_inv = np.linalg.inv(self.theta_cov)
        except np.linalg.LinAlgError:
            S0_inv = np.linalg.pinv(self.theta_cov)
        HtH = H.T @ H
        precision = S0_inv + (1 / (self.sig ** 2)) * HtH
        try:
            S_post = np.linalg.inv(precision)
        except np.linalg.LinAlgError:
            S_post = np.linalg.pinv(precision)
        rhs = (S0_inv @ self.theta_mean) + (1.0 / (self.sig ** 2)) * (H.T @ y)
        mu_post = S_post @ rhs
        self.theta_mean_post = mu_post
        self.theta_cov_post = S_post
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X).reshape(-1)
        H = self.basis(X)
        if self.theta_mean_post is None:
            return H @ self.theta_mean
        return H @ self.theta_mean_post

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.predict(X)

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X).reshape(-1)
        H = self.basis(X)
        if self.theta_cov_post is None:
            cov = self.theta_cov
            mean = self.theta_mean
        else:
            cov = self.theta_cov_post
            mean = self.theta_mean_post
        var = np.sum(H @ cov * H, axis=1) + (self.sig ** 2)
        std = np.sqrt(np.maximum(var, 0.0))
        return std

    def posterior_sample(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X).reshape(-1)
        H = self.basis(X)
        if self.theta_cov_post is None or self.theta_mean_post is None:
            mean = self.theta_mean
            cov = self.theta_cov
        else:
            mean = self.theta_mean_post
            cov = self.theta_cov_post
        try:
            L = np.linalg.cholesky(cov)
            z = np.random.randn(cov.shape[0])
            theta_samp = mean + L @ z
        except np.linalg.LinAlgError:
            theta_samp = np.random.multivariate_normal(mean, cov)
        return H @ theta_samp


class LinearRegression:
    def __init__(self, basis_functions: Callable):
        self.bases = basis_functions
        self.theta = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        X = np.asarray(X).reshape(-1)
        y = np.asarray(y).reshape(-1)
        H = self.bases(X)
        theta_hat, *_ = np.linalg.lstsq(H, y, rcond=None)
        self.theta = theta_hat
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X).reshape(-1)
        H = self.bases(X)
        return H @ self.theta

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.predict(X)


def plot_prediction_train_test_points(x_grid, true_x, true_y, pred_y, train_x=None, train_y=None, title=None):
    """
    Plot a prediction curve along with train and test points.
    """
    plt.figure()
    # plot train points if provided
    if train_x is not None and train_y is not None:
        plt.scatter(train_x, train_y, label='train points', marker='o', color='green')
    # plot test/true points
    plt.scatter(true_x, true_y, label='true test points', marker='o', color='red')
    plt.plot(x_grid, pred_y, label='prediction (mean)', lw=2)
    plt.title(title if title else '')
    plt.xlabel('hour')
    plt.ylabel('temperature [C]')
    plt.legend()
    plt.show()


def plot_prior_samples_and_mean(x_grid, mu, cov, basis_func, n_samples=5, title=None):
    H = basis_func(x_grid)
    mean_fn = H @ mu
    std = np.sqrt(np.maximum(np.sum(H @ cov * H, axis=1), 0.0))
    plt.figure()
    plt.fill_between(x_grid, mean_fn - std, mean_fn + std, alpha=0.3, label='prior ± std')
    try:
        L = np.linalg.cholesky(cov)
        for i in range(n_samples):
            z = np.random.randn(cov.shape[0])
            theta_samp = mu + L @ z
            plt.plot(x_grid, H @ theta_samp, lw=1)
    except np.linalg.LinAlgError:
        for i in range(n_samples):
            theta_samp = np.random.multivariate_normal(mu, cov)
            plt.plot(x_grid, H @ theta_samp, lw=1)
    plt.plot(x_grid, mean_fn, 'k', lw=2, label='prior mean')
    plt.title(title if title else 'Prior samples & mean')
    plt.xlabel('hour')
    plt.ylabel('temperature [C]')
    plt.legend()
    plt.show()


def plot_posterior_with_uncertainty(x_grid, train_hours, train_temps, blr: BayesianLinearRegression,
                                    n_samples=5, title=None, test_hours=None, test_temps=None):
    """
    Plot the posterior mean and uncertainty, training points and test points.
    """
    H = blr.basis(x_grid)
    mean_fn = blr.predict(x_grid)
    std = blr.predict_std(x_grid)
    plt.figure()
    plt.fill_between(x_grid, mean_fn - std, mean_fn + std, alpha=0.3, label='posterior ± std')
    for i in range(n_samples):
        try:
            L = np.linalg.cholesky(blr.theta_cov_post)
            z = np.random.randn(blr.theta_cov_post.shape[0])
            theta_samp = blr.theta_mean_post + L @ z
            plt.plot(x_grid, blr.basis(x_grid) @ theta_samp, lw=1)
        except Exception:
            theta_samp = np.random.multivariate_normal(blr.theta_mean_post, blr.theta_cov_post)
            plt.plot(x_grid, blr.basis(x_grid) @ theta_samp, lw=1)
    plt.plot(x_grid, mean_fn, 'k', lw=2, label='posterior mean')
    # plot training and test points
    plt.scatter(train_hours, train_temps, color='green', label='train points', marker='o')
    if test_hours is not None and test_temps is not None:
        plt.scatter(test_hours, test_temps, color='red', label='test points', marker='o')
    plt.title(title if title else 'Posterior mean & samples')
    plt.xlabel('hour')
    plt.ylabel('temperature [C]')
    plt.legend()
    plt.show()


def main():
    np.random.seed(0)
    # load the data for November 16 2024
    nov16 = np.load('nov162024.npy')
    nov16_hours = np.arange(0, 24, .5)
    train = nov16[:len(nov16) // 2]
    train_hours = nov16_hours[:len(nov16) // 2]
    test = nov16[len(nov16) // 2:]
    test_hours = nov16_hours[len(nov16) // 2:]

    # setup the model parameters
    degrees = [3, 7]

    # ----------------------------------------- Classical Linear Regression
    x_grid = np.linspace(0, 24, 400)
    for d in degrees:
        ln = LinearRegression(polynomial_basis_functions(d)).fit(train_hours, train)
        mse = np.mean((test - ln.predict(test_hours)) ** 2)
        print(f'Average squared error with LR and polynomial degree={d}: {mse:.4f}')

        # plot predicted curve (dense) with both train & true test points
        pred_grid = ln.predict(x_grid)
        plot_prediction_train_test_points(x_grid, test_hours, test, pred_grid,
                                          train_x=train_hours, train_y=train,
                                          title=f'Linear Regression of polynomial deg={d} (test MSE {mse:.4f})')

    # ----------------------------------------- Bayesian Linear Regression
    temps = np.load('jerus_daytemps.npy').astype(np.float64)
    hours = np.array([2, 5, 8, 11, 14, 17, 20, 23]).astype(np.float64)
    x = np.arange(0, 24, .1)

    sigma = 0.5
    degrees = [3, 7]
    freqs = [1, 2, 3]
    knots = [np.array([12]), np.array([8, 16]), np.array([6, 12, 18])]

    # ---------------------- polynomial basis functions
    for deg in degrees:
        pbf = polynomial_basis_functions(deg)
        mu, cov = learn_prior(hours, temps, pbf)
        blr = BayesianLinearRegression(mu, cov, sigma, pbf)
        plot_prior_samples_and_mean(x, mu, cov, pbf, n_samples=5, title=f'Prior (polynomial deg={deg})')
        blr.fit(train_hours, train)
        mse_post = np.mean((test - blr.predict(test_hours)) ** 2)
        plot_posterior_with_uncertainty(x, train_hours, train, blr, n_samples=5,
                                        title=f'Posterior (polynomial deg={deg}) MSE={mse_post:.4f}',
                                        test_hours=test_hours, test_temps=test)
        print(f'Bayesian LR (polynomial deg={deg}) MMSE test MSE: {mse_post:.4f}')

    # ---------------------- fourier basis functions
    for K in freqs:
        rbf = fourier_basis_functions(K)
        mu, cov = learn_prior(hours, temps, rbf)
        blr = BayesianLinearRegression(mu, cov, sigma, rbf)
        plot_prior_samples_and_mean(x, mu, cov, rbf, n_samples=5, title=f'Prior (fourier K={K})')
        blr.fit(train_hours, train)
        mse_post = np.mean((test - blr.predict(test_hours)) ** 2)
        plot_posterior_with_uncertainty(x, train_hours, train, blr, n_samples=5,
                                        title=f'Posterior (fourier K={K}) MSE={mse_post:.4f}',
                                        test_hours=test_hours, test_temps=test)
        print(f'Bayesian LR (fourier K={K}) MMSE test MSE: {mse_post:.4f}')

    # ---------------------- cubic regression splines
    for knots_set in knots:
        spline = spline_basis_functions(knots_set)
        mu, cov = learn_prior(hours, temps, spline)
        blr = BayesianLinearRegression(mu, cov, sigma, spline)
        plot_prior_samples_and_mean(x, mu, cov, spline, n_samples=5, title=f'Prior (spline knots={knots_set})')
        blr.fit(train_hours, train)
        mse_post = np.mean((test - blr.predict(test_hours)) ** 2)
        plot_posterior_with_uncertainty(x, train_hours, train, blr, n_samples=5,
                                        title=f'Posterior (spline knots={knots_set}) MSE={mse_post:.4f}',
                                        test_hours=test_hours, test_temps=test)
        print(f'Bayesian LR (spline knots={knots_set}) MMSE test MSE: {mse_post:.4f}')


if __name__ == '__main__':
    main()
