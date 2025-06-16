import os
import pylab as plt
import numpy as np
from scipy import signal
from adapt_utils import get_aligned_signals , display_and_save

def rls(x, d, N = 4, lmbd = 0.995, delta = 0.01):
  nIters = min(len(x),len(d)) - N
  lmbd_inv = 1/lmbd
  u = np.zeros(N)
  w = np.zeros(N)
  P = np.eye(N)*delta
  e = np.zeros(nIters)
  for n in range(nIters):
    u[1:] = u[:-1]
    u[0] = x[n]
    e_n = d[n] - np.dot(u, w)
    #
    do_adapt = x[n] != 0
    if do_adapt:
        r = np.dot(P, u)
        g = r / (lmbd + np.dot(u, r))
        w = w + e_n * g
        P = lmbd_inv*(P - np.outer(g, np.dot(u, P)))


    e[n] = e_n

  mic_est = np.hstack([e.flatten(), np.zeros(N, )])
  return mic_est




if __name__ == "__main__":
    import soundfile as sf
    from copy import copy
    import librosa
    inpath = 'C:/Users/dadab/projects/AEC/data/rec1/2'
    delay = 0
    mic_out, ref, sr = get_aligned_signals(inpath, delay = delay)

    mic = copy(mic_out)
    mic_sig_filtered = copy(mic_out)

    #####################################################
    # Get only the relevant part for adaptation
    first_ai_response = np.where(ref > 0)[0][0]
    mic = mic[first_ai_response:]
    ref = ref[first_ai_response:]

    # min_len = min(len(mic), len(ref))
    # mic = mic[:min_len]
    # ref = ref[:min_len]

    # Add talk to the mic during the ai response time
    if False:
        talk_len = len(mic)//4
        mic[talk_len:2*talk_len] += mic_out[:len(mic[talk_len:2*talk_len])]

    #####################################################

    N = 64
    filtered_mic = rls(ref, mic, N = N)

    display_and_save(mic, filtered_mic, ref, sr , inpath)

   #
   #  initial_power = np.mean(mic ** 2)
   #  final_power =  np.mean(filtered_mic ** 2)
   #  print(
   #      f"Echo suppression:  ,  {10 * np.log10(initial_power / final_power):.2f} dB  (energy reduction by factor of  {(initial_power / final_power):.2f}) ")
   #
   #  mic_sig_filtered[first_ai_response:len(mic_sig_filtered)] = filtered_mic[:len(mic_sig_filtered[first_ai_response:len(mic_sig_filtered)])]
   #  #mic_sig_filtered[first_ai_response:] = filtered_mic
   #  sf.write(os.path.join(inpath + '/mic_filtered_adaptive_rls.wav'),
   #                mic_sig_filtered,sr)
   #
   # # Display
   #  fig, axes = plt.subplots(2, 2, figsize=(15, 10))
   #
   #  axes[0, 0].plot(ref, label='ref')
   #  axes[0, 0].plot(mic, label='mic')
   #
   #  #plt.plot(mic_est, label='est mic')
   #  axes[0, 0].plot(filtered_mic, label='filtered mic  ')
   #  axes[0, 0].legend()
   #
   #  # Power spectral densities
   #  f, Pxx_echo = signal.welch(mic, fs=sr, nperseg=512)
   #  f, Pxx_cancelled = signal.welch(filtered_mic, fs=sr, nperseg=512)
   #
   #  axes[1, 0].semilogy(f, Pxx_echo, label='With echo', alpha=0.7)
   #  axes[1, 0].semilogy(f, Pxx_cancelled, label='Echo cancelled', alpha=0.7)
   #  axes[1, 0].set_xlabel('Frequency (Hz)')
   #  axes[1, 0].set_ylabel('PSD')
   #  axes[1, 0].set_title('Power Spectral Density')
   #  axes[1, 0].legend()
   #  axes[1, 0].grid(True)
   #
   #  # Coherence (quality metric)
   #  f, Cxy =  signal.coherence(mic, ref, fs=sr, nperseg=512)
   #
   #  axes[1, 1].plot(f, Cxy, 'r-', linewidth=2)
   #  axes[1, 1].set_xlabel('Frequency (Hz)')
   #  axes[1, 1].set_ylabel('Coherence')
   #  axes[1, 1].set_title('Coherence between mic  and ai ')
   #
   #
   #
   #  plt.show()
    # sf.write('e.wav', e, sr)
    # sf.write('y.wav', y, sr)
