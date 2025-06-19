import numpy as np
from adaptive_filters import AdaptiveRLSFilter , AdaptiveNLMSFilter


def run_sim(inpath):
    '''
    Simulation of delay
    :param inpath:
    :return:
    '''
    from adapt_utils import get_aligned_signals, display_and_save
    from copy import copy

    min_size_for_delay_estimation_sec = 0.25  # minimal number of time to use for valid delay estimation[sec]
    max_system_delay_sec = 0.5  # delay uncertainty
    #####################################################
    # Get the signals
    mic_out, ref, sr = get_aligned_signals(inpath, delay=0)
    batch_size = sr // 100  # Batch size ~= 10 msec

    mic = copy(mic_out)

    # Get only the relevant part for adaptation
    first_ai_response = np.where(ref > 0)[0][0]
    mic = mic[first_ai_response:]
    ref = ref[first_ai_response:]

    if False:
        # Add talk to the mic during the ai response time
        talk_len = len(mic) // 4
        mic[talk_len:2 * talk_len] += mic_out[:len(mic[talk_len:2 * talk_len])]

    #####################################################
    # Simulate an unknown delay
    system_delay_sec = 0.1  # simulate this delay
    system_delay_samples = int(system_delay_sec * sr)

    if system_delay_samples > 0:
        mic = mic[:-system_delay_samples]
        ref = ref[system_delay_samples:]
    else:
        pass

    ####################################################

    # Define the filter
    adaptfilter = AdaptiveRLSFilter(64, ref,
                                    min_size_for_delay_estimation=int(min_size_for_delay_estimation_sec * sr),
                                    delay_estimation_uncertainty_samples=int(max_system_delay_sec * sr))

    # Run the filter , feed it batch by batch of mic samples
    mic_filtered = np.zeros(len(mic))
    for b in range(0, len(mic), batch_size):
        e_batch = adaptfilter.process_batch(mic[b:b + batch_size], b)
        mic_filtered[b:b + batch_size] = e_batch

    display_and_save(mic, mic_filtered, ref, sr, inpath)


def debug_integration(inpath):
    import soundfile as sf
    from adapt_utils import display_and_save, get_aligned_signals

    # Read mic signal
    mic_sig, mic_sr = sf.read(inpath + '/mic_output.wav')
    # Read ai signal
    ai_sig, ai_sr = sf.read(inpath + '/original_ai.wav')


    min_size_for_delay_estimation_sec = 0.2  # minimal number of time to use for valid delay estimation[sec]
    max_system_delay_sec = 0.6  # delay uncertainty
    batch_size = (int)(mic_sr // (1000 / 80) ) # Batch size ~= 80 msec
    N  = (int)(len(mic_sig)// batch_size * batch_size)
    ai_sig = ai_sig[:N]
    mic_sig = mic_sig[:N]

    # Define the filter
    # adaptfilter = AdaptiveRLSFilter(64,
    #                                 int(min_size_for_delay_estimation_sec * mic_sr),
    #                                 int(max_system_delay_sec * mic_sr))
    adaptfilter = AdaptiveNLMSFilter(64,
                                    int(min_size_for_delay_estimation_sec * mic_sr),
                                    int(max_system_delay_sec * mic_sr))



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
    debug_integration('C:/Users/dadab/projects/AEC/data/integration3/2')

    # run_sim('C:/Users/dadab/projects/AEC/data/rec1/3')
