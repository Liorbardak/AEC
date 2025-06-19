import numpy as np
from adaptive_filters import AdaptiveRLSFilter , AdaptiveNLMSFilter
import soundfile as sf
from adapt_utils import display_and_save, get_aligned_signals
import cv2
import pylab as plt


def find_delay(inpath):
    # Read mic signal
    mic_sig, mic_sr = sf.read(inpath + '/mic_output.wav')
    # Read ai signal
    ai_sig, ai_sr = sf.read(inpath + '/original_ai.wav')

    mic_sig = mic_sig.astype(float)
    # Simple template matching for finding the shift between the signals
    range = np.min([len(mic_sig), mic_sr*3])
    signal = mic_sig[:range].reshape(1, -1).astype(np.float32)
    template = ai_sig[:range- mic_sr*2].reshape(1, -1).astype(np.float32)


    result = cv2.matchTemplate(signal, template, cv2.TM_CCOEFF_NORMED)
    best_alignment = np.argmax(result)

    plt.subplot(2, 1, 1)
    plt.plot(mic_sig,alpha = 0.7,label = 'mic')
    plt.plot(ai_sig,alpha = 0.7,label = 'ai')
    plt.title(f"delay {best_alignment} corr  {np.max(result) : .2f}")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(result.flatten())

    plt.show()


def debug_integration(inpath):


    # Read mic signal
    mic_sig, mic_sr = sf.read(inpath + '/mic_output.wav')
    # Read ai signal
    ai_sig, ai_sr = sf.read(inpath + '/original_ai.wav')


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
    for mic_ind in range(0, len(mic_sig), batch_size):

        mic_filtered[mic_ind:mic_ind + batch_size] = adaptfilter.process_batch(mic_sig[mic_ind:mic_ind + batch_size])

        # add more ai samples to the filter (arrive latter )
        if ai_ind+batch_size < len(ai_sig):
            adaptfilter.add_ai_response_batch(ai_sig[ai_ind:ai_ind + batch_size])
            ai_ind += batch_size

    display_and_save(mic_sig, mic_filtered, ai_sig, mic_sr, inpath)


if __name__ == "__main__":
    debug_integration('C:/Users/dadab/projects/AEC/data/a6/77')
    #debug_integration('C:/Users/dadab/projects/AEC/data/integration3/4')
    #find_delay('C:/Users/dadab/projects/AEC/data/a6/77')

