import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, csd
from scipy.fft import ifft, fft, fftfreq
from copy import copy
import librosa
from scipy.signal import lfilter, filtfilt , butter
# # Example: generate a system y = h * x (convolution)
# np.random.seed(0)
# x = np.random.randn(2048)
# true_h = np.array([0.5, -0.3, 0.1])
# y = np.convolve(x, true_h, mode='same')
x = np.zeros(100)
x[10] = 1



# y4 =lfilter(np.array([0,0,0,0,1,0,0]) , 1 , x)
# y1 = lfilter(np.array([0,1,0,0,0,0,0]) , 1 , x)
#
# plt.plot(x,label = 'x')
# plt.plot(y4,label = 'y4')
# plt.plot(y1,label = 'y1')
# plt.legend()
# plt.show()


inpath = 'C:/Users/dadab/projects/AEC/data/rec1/2'
mic_sig , mic_sr  = librosa.load(inpath + '/mic_output.wav', sr=None)
ai_sig , ai_sr  =  librosa.load(inpath + '/resampled_and_normalized_ai.wav', sr=None)
gp = 10
#x = ai_sig[43000+gp:]


x = copy(ai_sig[43000:])

#
# fs = 8000
# cutoff = 3000  # Hz
# order = 4
x = np.random.rand(10000,)
b, a = butter(4, 0.5, btype='low')

y = lfilter(b, a, x)

#y = copy(mic_sig[43000:-gp])
# y = copy(ai_sig[43000+gp:])
# x = copy(mic_sig[43000:-gp])

#y4 = np.convolve(x,  np.array([0,0,0,0,1,0,0]))


# Estimate PSD and CSD using Welch method
fs = 1.0  # sampling frequency (change if needed)

n = len(x)

f, Pxx = welch(x, fs=fs, nperseg=256)
_, Pxy = csd(x, y, fs=fs, nperseg=256)

# Transfer function estimate
H_f = Pxy / Pxx


# Interpolate to full FFT resolution
# To invert via IFFT, we need H(f) at regular frequency bins
f_full = fftfreq(n, d=1/fs)
Pxx_full = np.abs(fft(x))**2 / n
Pxy_full = fft(x) * np.conj(fft(y)) / n

# Avoid divide-by-zero
eps = 1e-10
H_f_full = Pxy_full / (Pxx_full + eps)

# Inverse FFT to get impulse response
h_est = np.real(ifft(H_f_full))

# Optional: truncate to match length of expected impulse response
h_est_truncated = h_est[:64]  # adjust length if needed

# Convolve estimated h with x
s_hat = np.convolve(x, h_est_truncated, mode='full')

plt.figure(figsize=(10, 4))
plt.plot(h_est_truncated)
# Plot
plt.figure(figsize=(10, 4))
plt.plot(f, np.abs(H_f), label='|H(f)|')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Estimated Transfer Function |H(f)| using Welch method')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(x, label = 'x')
plt.plot(y, label = 'y')
plt.plot(s_hat, label = 's_hat')
plt.legend()
plt.show()

plt.plot