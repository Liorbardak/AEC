import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, welch, csd, coherence, freqz
from scipy.optimize import minimize
import librosa
from copy import copy
# Generate test data
# np.random.seed(42)
# x = np.random.rand(10000, )
# b_true, a_true = butter(4, 0.5, btype='low')
# b_true = np.array([0,0,0,0,1,0,0,0])
# a_true = 1
# y = lfilter(b_true, a_true, x)
# # Add some noise to make it realistic
# y += 0.0001 * np.random.randn(len(y))
#
# print("True filter coefficients:")
# print(f"b = {b_true}")
# print(f"a = {a_true}")


inpath = 'C:/Users/dadab/projects/AEC/data/rec1/2'
stopp = np.inf
startp = 43000
inpath = 'C:/Users/dadab/projects/AEC/data/rec1/6'
startp = 80000
inpath = 'C:/Users/dadab/projects/AEC/data/rec1/7'
startp = 65000
stopp = 67000
mic_sig , mic_sr  = librosa.load(inpath + '/mic_output.wav', sr=None)
ai_sig , ai_sr  =  librosa.load(inpath + '/resampled_and_normalized_ai.wav', sr=None)
gp = 10
#x = ai_sig[43000+gp:]
if np.isinf(stopp):
    stopp = len(mic_sig)
gp = 10
x = copy(ai_sig[startp+gp:stopp])

b_true, a_true = butter(4, 0.5, btype='low')
# y = lfilter(b_true, a_true, x)
y = copy(mic_sig[startp:stopp-gp])

# Method 1: Frequency Domain Transfer Function Estimation
def estimate_transfer_function_freq(x, y, fs=1.0, nperseg=1024):
    """
    Estimate transfer function using cross-spectral density method
    H(ω) = Sxy(ω) / Sxx(ω)
    """
    # Compute cross power spectral density
    f, Sxy = csd(x, y, fs=fs, nperseg=nperseg)

    # Compute auto power spectral density of input
    f, Sxx = csd(x, x, fs=fs, nperseg=nperseg)

    # Transfer function estimate
    H = Sxy / Sxx

    # Coherence (quality metric)
    f, Cxy = coherence(x, y, fs=fs, nperseg=nperseg)

    return f, H, Cxy


# Method 2: Time Domain - Least Squares (ARX model)
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




# Apply estimation methods
print("\n" + "=" * 50)
print("TRANSFER FUNCTION ESTIMATION RESULTS")
print("=" * 50)

# Method 1: Frequency domain
f, H_est, Cxy = estimate_transfer_function_freq(x, y, fs=mic_sr)

# Method 2: ARX model estimation
b_arx, a_arx, theta = estimate_arx_model(x, y, na=1, nb=25, nk=0)

print(f"\nARX Model Estimation (na=4, nb=4, nk=1):")
print(f"Estimated b = {b_arx}")
print(f"Estimated a = {a_arx}")

# Method 3: Wiener filter
h_wiener = estimate_wiener_filter(x, y, filter_length=25)
print(f"\nWiener Filter (first 10 coefficients): {h_wiener[:20]}")

# Validation: Compare frequency responses
plt.figure(figsize=(15, 8))

# True transfer function
w_true, h_true = freqz(b_true, a_true, worN=512, fs=mic_sr)

# ARX model transfer function
w_arx, h_arx = freqz(b_arx, a_arx, worN=512, fs=mic_sr)

# Wiener filter transfer function (FIR)
w_wiener, h_wiener_freq = freqz(h_wiener, [1], worN=512, fs=mic_sr)

# Plot magnitude response
plt.subplot(2, 2, 1)
#plt.plot(w_true, 20 * np.log10(abs(h_true)), 'k-', linewidth=2, label='True')
plt.plot(f, 20 * np.log10(abs(H_est)), 'r--', alpha=0.7, label='Freq. Domain Est.')
plt.plot(w_arx, 20 * np.log10(abs(h_arx)), 'b:', label='ARX Model')
plt.plot(w_wiener, 20 * np.log10(abs(h_wiener_freq)), 'g-.', label='Wiener Filter')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('Transfer Function Magnitude')
plt.legend()
plt.grid(True)

# Plot phase response
plt.subplot(2, 2, 2)
#plt.plot(w_true, np.unwrap(np.angle(h_true)) * 180 / np.pi, 'k-', linewidth=2, label='True')
plt.plot(f, np.unwrap(np.angle(H_est)) * 180 / np.pi, 'r--', alpha=0.7, label='Freq. Domain Est.')
plt.plot(w_arx, np.unwrap(np.angle(h_arx)) * 180 / np.pi, 'b:', label='ARX Model')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (degrees)')
plt.title('Transfer Function Phase')
plt.legend()
plt.grid(True)

# Plot coherence
plt.subplot(2, 2, 3)
plt.plot(f, Cxy, 'r-', linewidth=2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Coherence')
plt.title('Coherence between x and y')
plt.grid(True)
plt.ylim(0, 1.1)

# Time domain validation
plt.subplot(2, 2, 4)
y_arx = lfilter(b_arx, a_arx, x)
y_wiener = lfilter(h_wiener, 1, x)

t = np.arange(len(x))
plt.plot(t, y[t], 'k-', label='True output')
plt.plot(t, y_arx[t], 'b--', alpha=0.7, label='ARX model')
plt.plot(t, y_wiener[t], 'g:', alpha=0.7, label='Wiener filter')
#plt.plot(t, y[t] - y_arx[t], 'y--', alpha=0.7, label='Out - ARX model')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.title('Time Domain Validation')
plt.legend()
plt.grid(True)


# Compute estimation errors
mse_arx = np.mean((y - y_arx) ** 2)
mse_wiener = np.mean((y - y_wiener) ** 2)

print(f"\nValidation MSE:")
print(f"Sig : { np.sqrt(np.mean(y ** 2)):.6f}")
print(f"ARX model: {np.sqrt(mse_arx):.6f}")
print(f"Wiener filter: {np.sqrt(mse_wiener):.6f}")

plt.tight_layout()
plt.show()

# Function for easy use
def estimate_transfer_function(x, y, method='arx', **kwargs):
    """
    Main function to estimate transfer function

    Parameters:
    x, y: input and output signals
    method: 'arx', 'freq', or 'wiener'
    **kwargs: method-specific parameters

    Returns:
    Estimated transfer function parameters
    """
    if method == 'arx':
        na = kwargs.get('na', 4)
        nb = kwargs.get('nb', 4)
        nk = kwargs.get('nk', 1)
        return estimate_arx_model(x, y, na, nb, nk)

    elif method == 'freq':
        fs = kwargs.get('fs', 1.0)
        nperseg = kwargs.get('nperseg', 1024)
        return estimate_transfer_function_freq(x, y, fs, nperseg)

    elif method == 'wiener':
        filter_length = kwargs.get('filter_length', 50)
        return estimate_wiener_filter(x, y, filter_length)

    else:
        raise ValueError("Method must be 'arx', 'freq', or 'wiener'")


# Example usage
# print(f"\nSimple usage examples:")
# print(f"b_est, a_est, _ = estimate_transfer_function(x, y, method='arx', na=4, nb=4)")
# print(f"f, H, coherence = estimate_transfer_function(x, y, method='freq', fs=2.0)")
# print(f"h_fir = estimate_transfer_function(x, y, method='wiener', filter_length=30)")

# # Show coefficient comparison
# print(f"\nCoefficient Comparison:")
# print(f"True b:      {b_true}")
# print(f"Estimated b: {b_arx}")
# print(f"True a:      {a_true}")
# print(f"Estimated a: {a_arx}")
# print(f"Mean coherence: {np.mean(Cxy):.3f}")
