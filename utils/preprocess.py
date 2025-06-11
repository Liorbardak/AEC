import numpy as np
import pylab as plt
from scipy.io import wavfile
import librosa
import os
from pathlib import Path
from scipy import signal
import cv2

import soundfile as sf

def align_ai_to_mic(inpath : str):
    '''
    Align recorded ai response to the recorded mic signal
    :param inpath:
    :return:
    '''

    mic_sr, mic_sig = wavfile.read(inpath + '/mic_output.wav')
    ai_sr, ai_sig = wavfile.read(inpath + '/original_ai.wav')
    mic_sig = mic_sig.astype(float)
    # Resample AI signal to the sample rate of the mic signal
    ai_sig = librosa.resample(ai_sig.astype(float), orig_sr=float(ai_sr), target_sr=float(mic_sr))
    if len(ai_sig)*1.2 > len(mic_sig) :
        print('bad signals - ai sig is too long')
        return
    # Simple template matching
    signal = mic_sig.reshape(1, -1).astype(np.float32)
    template = ai_sig.reshape(1, -1).astype(np.float32)
    result = cv2.matchTemplate(signal, template, cv2.TM_CCOEFF_NORMED)
    print(f" max corr {np.max(result)}")
    # Create  so-called aligned AI signal
    best_alignment = np.argmax(result)

    ai_sig_aligned = np.zeros(mic_sig.shape)

    ai_sig_aligned[best_alignment:best_alignment + ai_sig.shape[0]] = ai_sig.flatten()

    # scale AI signal - just for the display
    ai_sig_aligned_norm = ai_sig_aligned * np.sqrt( np.sum(mic_sig[best_alignment:best_alignment + ai_sig.shape[0]]**2) / np.sum(ai_sig**2))

    # Save wav file
    wavfile.write(os.path.join(inpath + '/resampled_ai.wav'), mic_sr, ai_sig_aligned.astype(np.int16))
    wavfile.write(os.path.join(inpath + '/resampled_and_normalized_ai.wav'), mic_sr, ai_sig_aligned_norm.astype(np.int16))
    # Display results
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    axes[0].plot(ai_sig_aligned_norm,label='ai')
    axes[0].plot(mic_sig, label='mic')
    axes[0].set_title('recorded signals')
    axes[0].legend()
    axes[1].plot(result.flatten(), label='result')
    axes[1].set_title('correlation')
    plt.savefig(os.path.join(inpath + '/comp.png'))
    plt.close('all')








if __name__ == "__main__":
    inpath = 'C:/Users/dadab/projects/AEC/data/rec1'
    for subdir in [item.name for item in Path(inpath).iterdir() if item.is_dir() and (item / 'mic_output.wav').is_file()]:
        print(subdir)
        align_ai_to_mic(os.path.join(inpath, subdir))

