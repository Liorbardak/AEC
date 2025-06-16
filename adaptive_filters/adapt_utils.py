import numpy as np
from typing import Tuple
import cv2
import os
from copy import copy
import soundfile as sf
import librosa
import pylab as plt
from scipy import signal

def get_aligned_signals(inpath: str, delay: int = 0):
    '''
    Get resampled and aligned mic and ai signals
    This part should be done in the online code
    :param inpath:
    :param delay: delay of the mic respect the ai signal
    :return: ai & mic signals , sample rate
    '''
    # Read mic signal
    mic_sig, mic_sr = sf.read(inpath + '/mic_output.wav')
    # Read ai signal
    ai_sig, ai_sr = sf.read(inpath + '/original_ai.wav')

    mic_sig = mic_sig.astype(float)
    # Resample AI signal to the sample rate of the mic signal
    ai_sig = librosa.resample(ai_sig.astype(float), orig_sr=float(ai_sr), target_sr=float(mic_sr))

    # Simple template matching for finding the shift between the signals
    signal = mic_sig.reshape(1, -1).astype(np.float32)
    template = ai_sig.reshape(1, -1).astype(np.float32)
    result = cv2.matchTemplate(signal, template, cv2.TM_CCOEFF_NORMED)
    print(f" max correlation between mic and ai response {np.max(result):.2f}")
    # Create  so-called aligned AI signal
    best_alignment = np.argmax(result)

    ai_sig_aligned = np.zeros(mic_sig.shape)

    ai_sig_aligned[best_alignment:best_alignment + ai_sig.shape[0]] = ai_sig.flatten()

    # Add delay to the mic signal respect the ai response - needed for proper casual adaptive filtering
    if delay > 0:
        mic_sig = mic_sig[:-delay]
        ai_sig_aligned = ai_sig_aligned[delay:]

    return mic_sig, ai_sig_aligned, mic_sr


def display_and_save(mic : np.array , filtered_mic : np.array  , ref : np.array , sr : float , outpath: str = None):
    '''
    Debug code for adaptive filter performance
    :param mic:
    :param mic_est:
    :param ref:
    :param sr: sample rate of the mic signal
    :param outpath:
    :return:
    '''

    # Check when ai starts
    first_ai_response = np.where(ref > 0)[0][0]


    initial_power = np.mean(mic[first_ai_response:] ** 2)
    final_power = np.mean(filtered_mic[first_ai_response:] ** 2)
    mic_sig_filtered = copy(mic)

    print(
        f"Echo suppression:  ,  {10 * np.log10(initial_power / final_power):.2f} dB  (energy reduction by factor of  {(initial_power / final_power):.2f}) ")


    # if outpath is not None:
    #     sf.write(outpath)

    # Display
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(ref, label='ref')
    axes[0, 0].plot(mic, label='mic')
    axes[0, 0].plot(filtered_mic, label='filtered mic  ')
    axes[0, 0].legend()
    axes[0, 0].set_title(f"Echo suppression:  ,  {10 * np.log10(initial_power / final_power):.2f} dB  (energy reduction by factor of  {(initial_power / final_power):.2f}) ")

    axes[0, 1].plot(mic[first_ai_response:], label='mic')
    axes[0, 1].plot(filtered_mic[first_ai_response:], label='est mic')
    axes[0, 1].legend()

    # Power spectral densities
    f, Pxx_echo = signal.welch(mic[first_ai_response:], fs=sr, nperseg=512)
    f, Pxx_cancelled = signal.welch(filtered_mic[first_ai_response:], fs=sr, nperseg=512)

    axes[1, 0].semilogy(f, Pxx_echo, label='With echo', alpha=0.7)
    axes[1, 0].semilogy(f, Pxx_cancelled, label='Echo cancelled', alpha=0.7)
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('PSD')
    axes[1, 0].set_title('Power Spectral Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Coherence (quality metric)
    f, Cxy = signal.coherence(mic[first_ai_response:], ref[first_ai_response:], fs=sr, nperseg=512)

    axes[1, 1].plot(f, Cxy, 'r-', linewidth=2)
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Coherence')
    axes[1, 1].set_title('Coherence between mic  and ai ')

    plt.show()