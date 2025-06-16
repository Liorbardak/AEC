import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
from collections import deque
import cv2

class AdaptiveFilter(ABC):
    """
     adaptive filter base
    """
    def __init__(self, filter_length: int, full_ai_response : np.array , delay_estimation_size : int = 0, delay_estimation_uncertainty_samples : int = 0):
        '''

        :param filter_length:
        :param full_ai_response: reference full ai response signal
        :param delay_estimation_size: delay estimation buffer size[samples]
        :param delay_estimation_uncertainty_samples: delay uncertainty[samples]
        '''
        self.filter_length = filter_length
        self.set_full_ai_response(full_ai_response)
        self.mic_hist_buffer =  deque([],maxlen=delay_estimation_size)
        if  self.mic_hist_buffer.maxlen == 0:
            self.delay = 0
        else:
            self.delay = None
        self.delay_estimation_uncertainty_samples = delay_estimation_uncertainty_samples


    def set_full_ai_response(self, full_ai_response : np.array ):
        self.full_ai_response = full_ai_response

    def reset(self):
        self.mic_hist_buffer.clear()
        if self.mic_hist_buffer.maxlen == 0:
            self.delay = 0
        else:
            self.delay = None

    @abstractmethod
    def filter_sample(self, x_new: float, d_sample: float, do_adapt: bool = True) :
        pass

    def estimate_delay(self, mic_batch):
        self.mic_hist_buffer.extend(mic_batch)

        if len(self.full_ai_response) < len(self.full_ai_response) <  self.delay_estimation_uncertainty_samples + self.mic_hist_buffer.__len__():
            print('not enough samples for delay estimation in the recorded ai response')
            return

        if (self.mic_hist_buffer.maxlen == self.mic_hist_buffer.__len__()) :
            # for now - do not estimate after buffer is full - do not use the cyclic quality TODO - sort this out
            return

        if(self.mic_hist_buffer.__len__() > self.delay_estimation_uncertainty_samples):

            signal = self.full_ai_response[:  self.delay_estimation_uncertainty_samples + self.mic_hist_buffer.__len__()].reshape(1, -1).astype(np.float32)
            template = np.array(list(self.mic_hist_buffer)).reshape(1, -1).astype(np.float32)
            result = cv2.matchTemplate(signal, template, cv2.TM_CCOEFF_NORMED).flatten()

            # Check that correlation is good enough
            maxcorri = np.argmax(result)
            maxcorr = result[maxcorri]

            if((maxcorr > 0.2 )  & (maxcorr / np.percentile(np.abs(result), 75) > 5)):
                self.delay = maxcorri
                print(f" max correlation between mic and ai response {np.max(result):.2f} @ {maxcorri}")

            #
            # # Check that the correlation is good
            # import pylab as plt
            # plt.figure()
            # plt.plot(template.flatten()/np.max(template))
            # plt.plot(signal.flatten() / np.max(signal))
            # plt.figure()
            # plt.plot(result.flatten())
            # plt.show()
            # pass


    def process_batch(self, mic_batch: np.array, index_in_reference : int):
        '''
        process a batch of microphone samples
        :param mic_batch:
        :param index_in_reference: the matching index in the reference buffer
        :return:
        '''

        self.estimate_delay(mic_batch)

        e_batch = mic_batch
        if self.delay is not None:
            d_batch = self.full_ai_response[self.delay + index_in_reference:self.delay+ index_in_reference+len(mic_batch)]
            if(d_batch.size == mic_batch.size):
                 e_batch =  self.filter_batch(d_batch, mic_batch)

        return e_batch

    def filter_batch(self, mic_batch: np.array, d_batch: np.array):
        '''
        process a batch of samples 
        '''
        e_batch = np.zeros_like(mic_batch)
        for i in np.arange(len(mic_batch)):
            e_batch[i] = self.filter_sample(mic_batch[i], d_batch[i], do_adapt=mic_batch[i] != 0)
    
        return e_batch


    

class AdaptiveRLSFilter(AdaptiveFilter):
    """
    rls adaptive filter
    """
    def __init__(self, filter_length: int, full_ai_response : np.array , delay_estimation_size : int = 0,delay_estimation_uncertainty_samples : int = 0,
                 lmbd: float = 0.995, delta: float = 0.01):
        '''
        :param filter_length:
        '''
        super().__init__(filter_length, full_ai_response, delay_estimation_size, delay_estimation_uncertainty_samples)
        self.lmbd = lmbd
        self.lmbd_inv = 1.0 / lmbd
        self.delta = delta

        # Reset filter
        self.reset()

    def reset(self):
        super().reset()
        self.u = np.zeros(self.filter_length)
        self.w = np.zeros(self.filter_length)
        self.P = np.eye(self.filter_length) * self.delta


    def filter_sample(self, x_new: float, d_sample: float, do_adapt: bool = True) -> Tuple:
        '''
        Process single sample
        :param x_new:   New input sample
        :param d_sample:    Desired signal sample
        :param do_adapt:  Whether to adapt coefficients
        :return:
        y : float
            Filter output
        e : float
            Error signal
        '''
        self.u[1:] = self.u[:-1]
        self.u[0] = x_new
        y = np.dot(self.u, self.w)
        e_n = d_sample - y
        if do_adapt:
            r = np.dot(self.P, self.u)
            g = r / (self.lmbd + np.dot(self.u, r))
            self.w = self.w + e_n * g
            self.P = self.lmbd_inv * (self.P - np.outer(g, np.dot(self.u, self.P)))
        return e_n

def run_sim(inpath):

    from adapt_utils import get_aligned_signals, display_and_save
    from copy import copy

    max_system_delay_sec = 0.5 #  delay uncertainty
    #####################################################
    # Get the signals
    mic_out, ref, sr = get_aligned_signals(inpath, delay=0)
    batch_size = sr // 100  # Batch size ~= 10 msec

    mic = copy(mic_out)


    # Get only the relevant part for adaptation
    first_ai_response = np.where(ref > 0)[0][0]
    mic = mic[first_ai_response:]
    ref = ref[first_ai_response:]


    # Add talk to the mic during the ai response time
    if False:
        talk_len = len(mic) // 4
        mic[talk_len:2 * talk_len] += mic_out[:len(mic[talk_len:2 * talk_len])]


    #####################################################
    # Simulate an unknown delay
    system_delay_sec = 0.1 # simulate this delay
    system_delay_samples = int(system_delay_sec * sr)

    if system_delay_samples > 0:
        ref = ref[:-system_delay_samples]
        mic = mic[system_delay_samples:]
    # elif  system_delay_samples < 0:
    #     ref = ref[:system_delay_samples]
    #     mic = mic[-system_delay_samples:]
    else:
        pass

    # make sure signals are exactly of batch size
    N = len(mic) // batch_size * batch_size
    mic = mic[:N]
    ref = ref[:N]

    ####################################################

    # Define the filter
    adaptfilter = AdaptiveRLSFilter(64, ref,
                                    delay_estimation_size = int(max_system_delay_sec*2 * sr) ,
                                    delay_estimation_uncertainty_samples = int(max_system_delay_sec * sr) )

    # Run the filter , feed it batch by batch of mic samples

    mic_filtered = np.zeros(len(mic))
    for b in range(0, len(mic), batch_size):
        e_batch = adaptfilter.process_batch(mic[b:b + batch_size], b)
        try:
            mic_filtered[b:b + batch_size] = e_batch
        except:
            pass

    #

    display_and_save(mic, mic_filtered, ref, sr, inpath)

if __name__ == "__main__":
    run_sim('C:/Users/dadab/projects/AEC/data/rec1/2')
