import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import soundfile as sf
from copy import copy

def analyze_performance(mic, reference, mic_filtered ,sr):
    """
    Analyze and plot echo cancellation performance
    """
    # Calculate performance
    attenuation  =  np.mean(mic ** 2) / np.mean(mic_filtered** 2)




    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Time domain signals
    t = np.arange(len(reference)) / sr

    axes[0, 0].plot(t, reference, label='reference', alpha=0.7)
    axes[0, 0].plot(t, mic, label='mic (with echo)', alpha=0.7)
    axes[0, 0].plot(t, mic_filtered, label='Echo cancelled', alpha=0.8)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title(f' Attenuation {attenuation:.2f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Echo vs estimate
    axes[0, 1].plot(t, mic, label='True echo', alpha=0.7)
    axes[0, 1].plot(t, mic_filtered, label='Echo estimate', alpha=0.7)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title('Echo Estimation')
    axes[0, 1].legend()
    axes[0, 1].grid(True)


    # Coherence (quality metric)
    f, Cxy =  signal.coherence(mic, ref, fs=sr, nperseg=512)

    axes[1, 0].plot(f, Cxy, 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Coherence')
    axes[1, 0].set_title('Coherence between mic  and ai ')



    # Power spectral densities
    f, Pxx_echo = signal.welch(mic, fs=sr, nperseg=512)
    f, Pxx_cancelled = signal.welch(mic_filtered, fs=sr, nperseg=512)

    axes[1, 1].semilogy(f, Pxx_echo, label='With echo', alpha=0.7)
    axes[1, 1].semilogy(f, Pxx_cancelled, label='Echo cancelled', alpha=0.7)
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('PSD')
    axes[1, 1].set_title('Power Spectral Density')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()



if __name__ == '__main__':
    i = 12
    inpath = f'C:/Users/dadab/projects/AEC/data/rec1_filtered/{i}'
    mic, sr = sf.read(inpath + '/mic_output.wav')
    ref, sr = sf.read(inpath + '/resampled_and_normalized_ai.wav')
    mic_filtered, sr = sf.read(inpath + '/mic_filtered.wav')


    # Get only the relevant part
    first_ai_response = np.where(ref > 0)[0][0]
    mic = mic[first_ai_response:]
    ref = ref[first_ai_response:]
    mic_filtered = mic_filtered[first_ai_response:]

    analyze_performance(mic, ref, mic_filtered, sr)
    plt.savefig(os.path.join(inpath + '/att.png'))
    plt.show()
    #
    # for i in [2]: # np.arange(1,12):
    #     try:
    #         inpath = f'C:/Users/dadab/projects/AEC/data/rec1/{i}'
    #         mic, sr = sf.read(inpath + '/mic_output.wav')
    #         ref, sr = sf.read(inpath + '/resampled_and_normalized_ai.wav')
    #         #mic_filtered, sr = sf.read(inpath + '/mic_filtered.wav')
    #         mic_filtered, sr = sf.read(inpath + '/mic_filtered_adaptive.wav')
    #
    #         # Get only the relevant part
    #         first_ai_response = np.where(ref > 0)[0][0]
    #         mic = mic[first_ai_response:]
    #         ref = ref[first_ai_response:]
    #         mic_filtered = mic_filtered[first_ai_response:]
    #
    #         analyze_performance(mic , ref,mic_filtered , sr)
    #         plt.savefig(os.path.join(inpath + '/att.png'))
    #         plt.show()
    #     except:
    #         pass

   # plt.show()