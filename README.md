# Bayesian Linear Regression: Temperature Prediction 🌡️📈

This repository contains the practical implementation of **Exercise 2: Bayesian Linear Regression** for the course *67564 - Bayesian Machine Learning*. 

The objective of this project is to predict the temperatures during the second half of a given day (November 16) using both **Classical Linear Regression** and **Bayesian Linear Regression**. The project explores how integrating historical data as a Gaussian prior affects predictive performance when compared to standard linear regression.

---

## 📖 Background & Objective

We are given a sequence of temperature recordings for the first half of a day (at 30-minute intervals). The task is to accurately predict the remaining 24 data points representing the second half of the day.

To do this, we use linear regression models mapped to different basis functions:
$$ y(t) = \theta^T h(t) + \eta $$

Where $\eta \sim \mathcal{N}(0, I\sigma^2)$ represents our noise assumption ($\sigma^2 = 0.25$). For the Bayesian approach, we leverage a dataset of historical November daily temperatures (`jerus_daytemps.npy`) to learn a Gaussian Prior over the weights $\theta \sim \mathcal{N}(\mu, \Sigma)$.

## ✨ Key Features & Implementations

1. **Classical Linear Regression (`LinearRegression`)**:
   - Fits weights using ordinary least squares (via `np.linalg.lstsq`).
   - Prone to overfitting without regularization, especially with high-degree polynomials.

2. **Bayesian Linear Regression (`BayesianLinearRegression`)**:
   - Computes the posterior distribution $p(\theta | \mathcal{D})$ by updating the learned Gaussian prior with the observed data from the first half of the day.
   - Calculates the Maximum A Posteriori / Minimum Mean Square Error (MMSE) estimator.
   - Provides confidence intervals for predictions by computing the standard deviation of the posterior predictive distribution.

3. **Basis Functions**:
   - **Polynomial Basis**: $h(t) = [1, (t/d), (t/d)^2, \dots, (t/d)^d]^T$. Scaled by the degree $d$ to maintain numerical stability and avoid massive floating-point overflows. Tested with $d \in \{3, 7\}$.
   - **Fourier Basis**: Oscillatory functions utilizing sines and cosines, ideal for cyclic behavior like daily temperatures. Tested with frequencies $K \in \{1, 2, 3\}$.
   - **Cubic Regression Splines**: Piece-wise cubic polynomials bound by continuity constraints on the second derivative. Defined over sets of specific knots.

---

## 📊 Results Summary

The Bayesian approach massively outperformed the Classical method in handling unseen data, primarily due to the regularization effect of the learned prior. 

**Mean Squared Error (MSE) on the Test Set (Second half of the day):**

| Method / Basis Function | Parameters | Test MSE |
| :--- | :--- | :--- |
| **Classical Linear Regression** | Polynomial $d=3$ | 351.2128 |
| **Classical Linear Regression** | Polynomial $d=7$ | 848707.5351 💥*(Massive Overfitting)* |
| **Bayesian Linear Regression** | Polynomial $d=3$ | 1.9082 |
| **Bayesian Linear Regression** | Polynomial $d=7$ | 0.3513 |
| **Bayesian Linear Regression** | Fourier $K=1$ | 0.8087 |
| **Bayesian Linear Regression** | Fourier $K=2$ | 0.3099 |
| **Bayesian Linear Regression** | Fourier $K=3$ | 0.3152 |
| **Bayesian Linear Regression** | Spline knots=`[12]` | 0.4066 |
| **Bayesian Linear Regression** | Spline knots=`[8, 16]` | 1.3218 |
| **Bayesian Linear Regression** | Spline knots=`[6, 12, 18]` | **0.2468** 🏆 *(Best)* |

### Insights:
*   Standard linear regression with a 7th-degree polynomial spectacularly overfits the first half of the day, causing enormous errors in the second half.
*   Bayesian Linear Regression uses historical temperature trends to keep predictions grounded, heavily mitigating extreme divergence.
*   **Cubic Splines with 3 knots** achieved the highest accuracy, closely followed by the Fourier basis functions, which inherently model the rise-and-fall nature of daily temperatures well.

---

## 📁 Repository Structure

```text
├── ex2.py                   # Main script containing model classes and plotting logic
├── ex2_utils.py             # Utility functions for loading data and examples
├── jerus_daytemps.npy       # Historical November temperature dataset (for learning the prior)
├── nov162024.npy            # Test data (Temperatures for Nov 16, 2024)
├── Ex2.pdf                  # Official project instructions and theoretical background
├── plots from practical part.pdf # Output graphs showing prior/posterior means, samples, and MSE
└── README.md                # This file
```

---

## 🚀 How to Run

### Prerequisites
Make sure you have Python 3 installed along with the following packages:
```bash
pip install numpy matplotlib scipy scikit-image
```

### Execution
Simply run the main script. The script will sequentially train the models, calculate the MSE for each, and display the corresponding Matplotlib figures.
```bash
python ex2.py
```
*(Note: Close each plot window as it appears to let the script progress to the next model).*
