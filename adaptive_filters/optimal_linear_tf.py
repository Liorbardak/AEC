import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter, welch, csd, coherence, freqz
import os
from scipy.optimize import minimize
import librosa
from copy import copy
from scipy.io import wavfile
from utils.transfer_function_estimation import estimate_transfer_function, apply_transfer_function , estimate_arx_model , estimate_wiener_filter
def simple_noise_reduction(inpath):

    # mic_sig, mic_sr = librosa.load(inpath + '/mic_output.wav', sr=None)
    # ai_sig, ai_sr = librosa.load(inpath + '/resampled_and_normalized_ai.wav', sr=None)
    mic_sr, mic_sig = wavfile.read(inpath + '/mic_output.wav')
    ai_sr, ai_sig = wavfile.read(inpath + '/resampled_and_normalized_ai.wav')
    mic_sig = mic_sig.astype('float32')
    ai_sig = ai_sig.astype('float32')

    # add a small delay (make this filter casual)
    #gp = 5
    # mic_sig = mic_sig[:-gp]
    # ai_sig = ai_sig[gp:]
    # ai_sig = ai_sig[:-gp]
    # mic_sig = mic_sig[gp:]

    mic_sig_filtered = copy(mic_sig)
    first_ai_response = np.where(ai_sig > 0)[0][0]

    # Go over time segments - if there is active AI segment - try to estimate the filter
    seg_len_samples = int(0.33 * mic_sr)  # segment length
    for si in np.arange(0, mic_sig.shape[0] , seg_len_samples):
        seg_ai_sig = ai_sig[si:si+seg_len_samples]
        seg_mic_sig = mic_sig[si:si + seg_len_samples]
        if(np.sqrt(np.mean(seg_ai_sig**2)) > 0):

            # b_arx, a_arx, theta = estimate_arx_model(seg_ai_sig, seg_mic_sig, na=1, nb=15, nk=0)
            # mic_est = lfilter(b_arx, a_arx, seg_ai_sig)


            freq, H = estimate_transfer_function(seg_ai_sig, seg_mic_sig,fs =mic_sr )
            mic_est = apply_transfer_function(seg_ai_sig, H, freq,fs =mic_sr )



            mic_sig_filtered[si:si + seg_len_samples] = seg_mic_sig - mic_est

            # plt.figure()
            # plt.plot(seg_ai_sig,label = 'ai')
            # plt.plot(seg_mic_sig,label = 'mic')
            # plt.plot(mic_est, label='mic est')
            # plt.legend()
            # print(f" attenuation  {np.mean(seg_mic_sig**2) / np.mean((seg_mic_sig-mic_est)**2)} ")

    print(f"overall attenuation  {np.mean(mic_sig[first_ai_response:]**2) / np.mean((mic_sig_filtered[first_ai_response:])**2)} ")


    wavfile.write(os.path.join(inpath + '/mic_filtered.wav'), mic_sr,
                  mic_sig_filtered.astype(np.int16))
    # plt.figure()
    # plt.plot(mic_sig, label = 'mic')
    # plt.plot(mic_sig_filtered, label='mic filt')
    # plt.legend()
    #

  # Display
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    mic= mic_sig[first_ai_response:]
    filtered_mic = mic_sig_filtered[first_ai_response:]
    ref = ai_sig[first_ai_response:]
    sr = mic_sr

    axes[0, 0].plot(ref, label='ref')
    axes[0, 0].plot(mic, label='mic')

    #plt.plot(mic_est, label='est mic')
    axes[0, 0].plot(filtered_mic, label='filtered mic  ')
    axes[0, 0].legend()

    # Power spectral densities
    f, Pxx_echo = signal.welch(mic, fs=sr, nperseg=512)
    f, Pxx_cancelled = signal.welch(filtered_mic, fs=sr, nperseg=512)

    axes[1, 0].semilogy(f, Pxx_echo, label='With echo', alpha=0.7)
    axes[1, 0].semilogy(f, Pxx_cancelled, label='Echo cancelled', alpha=0.7)
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('PSD')
    axes[1, 0].set_title('Power Spectral Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Coherence (quality metric)
    f, Cxy =  signal.coherence(mic, ref, fs=sr, nperseg=512)

    axes[1, 1].plot(f, Cxy, 'r-', linewidth=2)
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Coherence')
    axes[1, 1].set_title('Coherence between mic  and ai ')



    plt.show()

if __name__ == "__main__":
    inpath = 'C:/Users/dadab/projects/AEC/data/rec1/12'
    simple_noise_reduction(inpath)
