import numpy as np
import soundfile as sf
import cv2
import pylab as plt
import librosa
import os
from scipy import signal

from adapt_utils import display_and_save, get_aligned_signals
from adaptive_filters import AdaptiveRLSFilter , AdaptiveNLMSFilter

def disp_integration(inpath , name1 ='echo', name2 ='original_output' , lastsamp = None):
    # Read mic signal
    name1_sig, sr = sf.read(inpath + '/' + name1 + '.wav')
    # Read ai signal
    name2_sig, sr2 = sf.read(inpath +  '/' + name2 + '.wav')
    if sr != sr2:
        name2_sig = librosa.resample(name2_sig.astype(float), orig_sr=float(sr2), target_sr=float(sr))
    name1_sig = name1_sig.astype(float)

    if lastsamp is not None:
        name1_sig = name1_sig[:lastsamp]
        name2_sig = name2_sig[:lastsamp]
    plt.figure()
    plt.plot(name1_sig,alpha = 0.7,label = name1)
    plt.plot(name2_sig,alpha = 0.7,label = name2)
    plt.legend()

    plt.show()

def find_delay(inpath , name1 ='mic_output', name2 ='original_ai' , lastsamp = None):
    # Find delay between ai & mic
    # Read mic signal
    name1_sig, sr = sf.read(inpath + '/' + name1 + '.wav')
    # Read ai signal
    name2_sig, sr2 = sf.read(inpath +  '/' + name2 + '.wav')
    if sr != sr2:
        name2_sig = librosa.resample(name2_sig.astype(float), orig_sr=float(sr2), target_sr=float(sr))
    name1_sig = name1_sig.astype(float)

    if lastsamp is not None:
        name1_sig = name1_sig[:lastsamp]
        name2_sig = name2_sig[:lastsamp]

    # Simple template matching for finding the shift between the signals
    range = np.min([len(name1_sig), sr*5])
    signal = name1_sig[:range].reshape(1, -1).astype(np.float32)
    template = name2_sig[:range- (int)(sr*1.5)].reshape(1, -1).astype(np.float32)


    result = cv2.matchTemplate(signal, template, cv2.TM_CCOEFF_NORMED)
    best_alignment = np.argmax(result)

    name1_sig = name1_sig[best_alignment:]
    plt.subplot(2, 1, 1)
    plt.plot(name1_sig,alpha = 0.7,label = name1)
    plt.plot(name2_sig,alpha = 0.7,label = name2)
    plt.title(f"delay {best_alignment} corr  {np.max(result) : .2f}")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(result.flatten())

    plt.show()


def debug_integration(inpath ,name1 = 'mic_output.wav', name2 = 'original_ai.wav',  lastsamp = None):


    # Read mic signal
    mic_sig, mic_sr = sf.read(inpath + '/' + name1)
    # Read AI signal
    ai_sig, ai_sr = sf.read(inpath + '/' + name2)
    if lastsamp is not None:
        ai_sig = ai_sig[:lastsamp]
        mic_sig = mic_sig[:lastsamp]

    # ai_sig = ai_sig[:-10000]
    # mic_sig = mic_sig[10000:]

    min_size_for_delay_estimation_sec = 0.2  # minimal number of time to use for valid delay estimation[sec]
    max_system_delay_sec = 0.8 # delay uncertainty
    batch_size = (int)(mic_sr // (1000 / 80) ) # Batch size ~= 80 msec
    N  = (int)(len(mic_sig)// batch_size * batch_size)


    ai_sig = ai_sig[:N]
    mic_sig = mic_sig[:N]
    # Reduce signal size
    # ai_sig = ai_sig[:mic_sr*3]
    # mic_sig = mic_sig[:mic_sr*3]


    # Define the filter
    adaptfilter = AdaptiveRLSFilter(64,
                                    int(min_size_for_delay_estimation_sec * mic_sr),
                                    int(max_system_delay_sec * mic_sr))
    # adaptfilter = AdaptiveNLMSFilter(64,
    #                                 int(min_size_for_delay_estimation_sec * mic_sr),
    #                                 int(max_system_delay_sec * mic_sr))



    # Add few ai response samples to the filter
    #for ai_ind in range(0,batch_size*10, batch_size):
    for ai_ind in range(0, len(ai_sig), batch_size):
        adaptfilter.add_ai_response_batch(ai_sig[ai_ind:ai_ind + batch_size])
    ai_ind += batch_size

    # Run the filter , feed it batch by batch of mic samples
    mic_filtered = np.zeros(len(mic_sig))
    AI_energy_est = np.zeros(len(mic_sig))
    mic_energy_est = []
    for mic_ind in range(0, len(mic_sig), batch_size):

        mic_filtered[mic_ind:mic_ind + batch_size], AI_energy = adaptfilter.process_batch(mic_sig[mic_ind:mic_ind + batch_size])
        AI_energy_est[mic_ind:mic_ind + batch_size] = AI_energy
        mic_energy_est.append(np.max([np.mean(mic_filtered[mic_ind:mic_ind + batch_size]**2)-AI_energy,0]))

        # add more AI samples to the filter (arrive latter )
        if ai_ind+batch_size < len(ai_sig):
            adaptfilter.add_ai_response_batch(ai_sig[ai_ind:ai_ind + batch_size])
            ai_ind += batch_size
    est_mic = mic_sig - mic_filtered


    mic_energy_est_sqrt = signal.lfilter(np.ones(3,)/3, 1, np.sqrt(mic_energy_est))
    mic_energy_est = np.zeros(len(mic_sig))
    i = 0
    for mic_ind in range(0, len(mic_sig), batch_size):
        mic_energy_est[mic_ind:mic_ind + batch_size]= mic_energy_est_sqrt[i]**2
        i = i + 1
    adaptfilter.save('d')

    display_and_save(mic_sig, mic_filtered,est_mic,  ai_sig, mic_sr, AI_energy_est ,mic_energy_est ,  inpath ,   delay=  adaptfilter.delay)


if __name__ == "__main__":
    #debug_integration('C:/Users/dadab/projects/AEC/data/a10' ,name1 = 'f_7_mic_output.wav' ,name2 = 'f_7_original_ai.wav')  # ,
    #debug_integration('C:/Users/dadab/projects/AEC/data/a8')# ,lastsamp = 40000)
    #debug_integration('C:/Users/dadab/projects/AEC/data/a7')# ,
    #disp_integration('C:/Users/dadab/projects/AEC/data/a8')
    find_delay('C:/Users/dadab/projects/AEC/data/a10', 'f_7_mic_output' , 'f_7_original_ai' ,lastsamp = 7*16000)

    #debug_integration('C:/Users/dadab/projects/AEC/data/integration3/4')

    #find_delay('C:/Users/dadab/projects/AEC/data/integration3/4')

    plt.show()
