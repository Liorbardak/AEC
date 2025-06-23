import numpy as np
from typing import Tuple
import cv2
import os
from copy import copy
import soundfile as sf
import librosa
import pylab as plt
from scipy import signal

def get_aligned_signals(inpath: str, delay: int = 0 , display :bool = False):
    '''
    Get resampled and aligned mic and ai signals
    This part should be done in the online code
    :param inpath:
    :param delay: delay of the mic respect the ai signal
    :param display: display
    :return: ai & mic signals , sample rate
    '''
    # Read mic signal
    mic_sig, mic_sr = sf.read(inpath + '/mic_output.wav')
    # Read ai signal
    ai_sig, ai_sr = sf.read(inpath + '/original_ai.wav')

    mic_sig = mic_sig.astype(float)
    # Resample AI signal to the sample rate of the mic signal
    ai_sig = librosa.resample(ai_sig.astype(float), orig_sr=float(ai_sr), target_sr=float(mic_sr))

    if len(mic_sig) > len(ai_sig):
        # Simple template matching for finding the shift between the signals
        signal = mic_sig.reshape(1, -1).astype(np.float32)
        template = ai_sig.reshape(1, -1).astype(np.float32)

        result = cv2.matchTemplate(signal, template, cv2.TM_CCOEFF_NORMED)
        best_alignment = np.argmax(result)
        print(f" max correlation between mic and ai response {np.max(result):.2f} , {best_alignment}  ")
        # Create  so-called aligned AI signal

        ai_sig_aligned = np.zeros(mic_sig.shape)
        ai_sig_aligned[best_alignment:best_alignment + ai_sig.shape[0]] = ai_sig.flatten()
    else:
        # Simple template matching for finding the shift between the signals
        template = mic_sig.reshape(1, -1).astype(np.float32)
        signal = ai_sig.reshape(1, -1).astype(np.float32)

        result = cv2.matchTemplate(signal, template, cv2.TM_CCOEFF_NORMED)
        best_alignment = np.argmax(result)

        print(f" max correlation between mic and ai response {np.max(result):.2f} , {best_alignment}  ")
        # Create  so-called aligned AI signal

        mic_sig_aligned = np.zeros(ai_sig.shape)
        mic_sig_aligned[best_alignment:best_alignment + mic_sig.shape[0]] = mic_sig.flatten()

        ai_sig_aligned = ai_sig
        mic_sig = mic_sig_aligned



    if (display):
        plt.subplot(2,1,1)
        plt.plot(mic_sig,alpha=0.7 ,label = 'mic')
        plt.plot(ai_sig_aligned,alpha=0.7,label = 'aligned ai')
        plt.legend()
        plt.title(f" correlation {np.max(result):.2f} , delay [samples] {best_alignment} [sec] {best_alignment / mic_sr : .2f} ")
        plt.subplot(2, 1, 2)
        plt.plot(result.flatten())
        plt.title("correlation")
        plt.show()


    # Add delay to the mic signal respect the ai response - needed for proper casual adaptive filtering
    if delay > 0:
        mic_sig = mic_sig[:-delay]
        ai_sig_aligned = ai_sig_aligned[delay:]

    return mic_sig, ai_sig_aligned, mic_sr


def display_and_save(mic : np.array , filtered_mic : np.array, est_mic : np.array  , ref : np.array , sr : float , AI_energy_est :np.array,mic_energy_est : np.array, outpath: str = None , delay :int  = None):
    '''
    Debug code for adaptive filter performance
    :param mic:
    :param filtered_mic
    :param mic_est:
    :param ref:
    :param sr: sample rate of the mic signal
    :param outpath:
    :return:
    '''

    if outpath is not None:
        sf.write(os.path.join(outpath + '/mic_filtered_adaptive.wav'),
                 filtered_mic, sr)

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
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[1, 0].sharex(axes[0, 0])
    axes[1, 1].sharex(axes[0, 0])

    axes[0, 0].plot(ref, label='ref')
    if delay is not None:
        axes[0, 0].plot(mic[delay:], label='mic')
        axes[0, 0].plot(filtered_mic[delay:], label='filtered mic  ')
    else:
        axes[0, 0].plot(mic, label='mic')
        axes[0, 0].plot(filtered_mic, label='filtered mic  ')
    axes[0, 0].legend()
    axes[0, 0].set_title(f"Echo suppression:  ,  {10 * np.log10(initial_power / final_power):.2f} dB  (energy reduction by factor of  {(initial_power / final_power):.2f}) ")

    flt = np.ones(100, )/100
    mic_energy = signal.filtfilt(flt, 1, mic[first_ai_response:]**2)
    filtered_mic_energy = signal.filtfilt(flt, 1, filtered_mic[first_ai_response:] ** 2)
    est_mic_energy = signal.filtfilt(flt, 1, est_mic[first_ai_response:] ** 2)

    if delay is not None:
        axes[1, 0].plot(mic_energy[delay:], label='mic energy ')
        axes[1, 0].plot(filtered_mic_energy[delay:], label='filtered_mic energy')
        axes[1, 0].plot(est_mic_energy[delay:], label='est_mic energy')
        axes[1, 0].plot(mic_energy_est[delay:] , alpha = 0.4, label='mic energy ')
        axes[1, 0].plot(AI_energy_est[delay:], alpha=0.4, label='est_mic from code ')

    else:
        axes[1, 0].plot(mic_energy, label='mic energy ')
        axes[1, 0].plot(filtered_mic_energy, label='filtered_mic energy')
        axes[1, 0].plot(est_mic_energy, label='est_mic energy')
        axes[1, 0].plot(mic_energy_est, alpha=0.4, label='mic energy ')
        axes[1, 0].plot(AI_energy_est, alpha=0.4, label='est_mic energy from code ')

    axes[1, 0].legend()

    # axes[1, 0].plot(mic[first_ai_response:], label='mic' )
    # axes[1, 0].plot(filtered_mic[first_ai_response:], label='filtered_mic')
    # axes[1,0].plot(est_mic[first_ai_response:], label='est_mic mic  ')
    # axes[1, 0].legend()

    # Power spectral densities
    f, Pxx_echo = signal.welch(mic[first_ai_response:], fs=sr, nperseg=512)
    f, Pxx_cancelled = signal.welch(filtered_mic[first_ai_response:], fs=sr, nperseg=512)

    axes[0, 1].semilogy(f, Pxx_echo, label='With echo', alpha=0.7)
    axes[0, 1].semilogy(f, Pxx_cancelled, label='Echo cancelled', alpha=0.7)
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('PSD')
    axes[0, 1].set_title('Power Spectral Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # # Coherence (quality metric)
    # f, Cxy = signal.coherence(mic[first_ai_response:], ref[first_ai_response:], fs=sr, nperseg=512)
    #
    # axes[1, 1].plot(f, Cxy, 'r-', linewidth=2)
    # axes[1, 1].set_xlabel('Frequency (Hz)')
    # axes[1, 1].set_ylabel('Coherence')
    # axes[1, 1].set_title('Coherence between mic  and ai ')
    #plt.tight_layout()
    if delay is not None:
        #axes[1, 1].plot(10*np.log10(mic_energy[delay:]+1e-10 ), label='original mic energy ')
        axes[1,1].plot(10*np.log10(filtered_mic_energy[delay:]+1e-10 ), label='echo energy')
        #axes[1, 1].plot(10*np.log10(est_mic_energy[delay:]+1e-10 ), label='est_mic energy')
        axes[1, 1].plot(10*np.log10(mic_energy_est[delay:]+1e-10 ) , label='energy after subtruction  ')
        #axes[1, 1].plot(10*np.log10(AI_energy_est[delay:]+1e-10 ), alpha=0.4, label='mic energy  from code ')

    else:
        #axes[1, 1].plot(10*np.log10(mic_energy+1e-10 ), label='mic energy ')
        axes[1, 1].plot(10*np.log10(filtered_mic_energy+1e-10 ), label='echo energy')
        #axes[1, 1].plot(10*np.log10(est_mic_energy+1e-10 ), label='est_mic energy')
        axes[1, 1].plot(10*np.log10(mic_energy_est+1e-10 ),  label=' energy after subtruction' )
        #axes[1, 1].plot(10*np.log10(AI_energy_est+1e-10 ), alpha=0.4, label='est_mic energy from code ')
    axes[1, 1].legend()


if __name__ == '__main__':
    mic_sig, ai_sig_aligned, mic_sr =    get_aligned_signals('C:/Users/dadab/projects/AEC/data/integration4')