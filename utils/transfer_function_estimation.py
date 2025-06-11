
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy import signal

def estimate_transfer_function(x, y, fs=1.0, nperseg=None, method='welch'):
    """
    Estimate transfer function H(ω) = Sxy(ω) / Sxx(ω) between two signals

    Parameters:
    -----------
    x : array_like
        Input signal
    y : array_like
        Output signal
    fs : float
        Sampling frequency (default: 1.0)
    nperseg : int or None
        Length of each segment for Welch's method
    method : str
        Method to use ('welch' or 'fft')

    Returns:
    --------
    freq : ndarray
        Frequency array
    H : ndarray
        Complex transfer function H(ω)
    """

    if method == 'welch':
        # Use Welch's method for better noise reduction
        if nperseg is None:
            nperseg = min(len(x )//4, 256)

        # Cross power spectral density Sxy
        freq, Sxy = signal.csd(x, y, fs=fs, nperseg=nperseg)

        # Auto power spectral density Sxx
        _, Sxx = signal.welch(x, fs=fs, nperseg=nperseg)

        # Transfer function H(ω) = Sxy(ω) / Sxx(ω)
        # Add small regularization to avoid division by zero
        eps = 1e-10 * np.max(np.abs(Sxx))
        H = Sxy / (Sxx + eps)

    elif method == 'fft':
        # Direct FFT method
        X = fft(x)
        Y = fft(y)

        # Cross spectral density
        Sxy = Y * np.conj(X)

        # Auto spectral density
        Sxx = X * np.conj(X)

        # Transfer function
        eps = 1e-10 * np.max(np.abs(Sxx))
        H = Sxy / (Sxx + eps)

        freq = fftfreq(len(x), 1/ fs)

    return freq, H


def apply_transfer_function(x, H, freq, fs=1.0):
    """
    Apply transfer function to input signal

    Parameters:
    -----------
    x : array_like
        Input signal
    H : array_like
        Transfer function (complex)
    freq : array_like
        Frequency array corresponding to H
    fs : float
        Sampling frequency

    Returns:
    --------
    y_est : ndarray
        Estimated output signal
    """

    # FFT of input
    X = fft(x)
    freq_full = fftfreq(len(x), 1 / fs)

    # Interpolate transfer function to match full frequency grid
    if len(H) != len(X):
        H_interp = np.interp(freq_full, freq, np.real(H)) + \
                   1j * np.interp(freq_full, freq, np.imag(H))
    else:
        H_interp = H

    # Apply transfer function
    Y_est = X * H_interp

    # Inverse FFT to get time domain signal
    y_est = np.real(np.fft.ifft(Y_est))

    return y_est


def estimate_arx_model(x, y, na=4, nb=4, nk=1):
    """
    Estimate ARX model: A(q)y(t) = B(q)x(t-nk) + e(t)

    Parameters:
    na: order of A polynomial (denominator)
    nb: order of B polynomial (numerator)
    nk: delay (number of samples)
    """
    N = len(x)

    # Build regression matrix
    max_order = max(na, nb + nk)
    phi = np.zeros((N - max_order, na + nb))
    y_reg = y[max_order:]

    # Past outputs (for A polynomial)
    for i in range(na):
        phi[:, i] = -y[max_order - 1 - i:-1 - i]

    # Past inputs (for B polynomial)
    for i in range(nb):
        phi[:, na + i] = x[max_order - nk - i:-nk - i if nk + i > 0 else None]

    # Least squares solution
    theta = np.linalg.lstsq(phi, y_reg, rcond=None)[0]

    # Extract A and B polynomials
    a_est = np.concatenate(([1], theta[:na]))
    b_est = theta[na:]

    return b_est, a_est, theta


# Method 3: Wiener Filter Approach
def estimate_wiener_filter(x, y, filter_length=50):
    """
    Estimate FIR filter using Wiener-Hopf equations
    For causal filter: R_xx * h = r_xy
    """
    N = len(x)

    # Compute autocorrelation matrix R_xx (Toeplitz)
    # R_xx[i,j] = E[x(n-i)x(n-j)] for i,j = 0,1,...,L-1
    R_xx = np.zeros((filter_length, filter_length))
    for i in range(filter_length):
        for j in range(filter_length):
            lag = abs(i - j)
            if lag == 0:
                R_xx[i, j] = np.mean(x ** 2)
            else:
                # Autocorrelation with proper lag
                R_xx[i, j] = np.mean(x[lag:] * x[:-lag])

    # Compute cross-correlation vector r_xy
    # r_xy[i] = E[x(n-i)y(n)] for i = 0,1,...,L-1
    r_xy = np.zeros(filter_length)
    for i in range(filter_length):
        if i == 0:
            r_xy[i] = np.mean(x * y)
        else:
            # Cross-correlation: x delayed by i samples
            r_xy[i] = np.mean(x[:-i] * y[i:])

    # Solve Wiener-Hopf equations: R_xx * h = r_xy
    try:
        h_wiener = np.linalg.solve(R_xx, r_xy)
    except np.linalg.LinAlgError:
        # If singular, use pseudo-inverse
        h_wiener = np.linalg.pinv(R_xx) @ r_xy

    return h_wiener

