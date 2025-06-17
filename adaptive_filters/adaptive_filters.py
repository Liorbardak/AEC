import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
import cv2

class AdaptiveFilter(ABC):
    """
     adaptive filter base
    """
    def __init__(self, filter_length: int, full_ai_response : np.array , min_size_for_delay_estimation : int = 0, delay_estimation_uncertainty_samples : int = 0):
        '''

        :param filter_length: adaptive filter length
        :param full_ai_response: the reference  ai response signal
        :param min_size_for_delay_estimation:minimal number of samples to match for delay estimation [samples]
        :param delay_estimation_uncertainty_samples: delay uncertainty[samples] , if = 0 - dont estimate delay
        '''

        self.filter_length = filter_length
        self.set_full_ai_response(full_ai_response)

        self.delay_correlation_accumulator = None
        self.min_size_for_delay_estimation = min_size_for_delay_estimation

        self.delay_estimation_uncertainty_samples = delay_estimation_uncertainty_samples
        self.additional_delay = 16 # add a bit more delay to the ai response , improve the attenuation of the casual filter

        if self.delay_estimation_uncertainty_samples == 0:
            # delay estimation is disabled
            self.delay = 0
        else:
            # delay estimation is enabled - need to estimate the delay latter
            self.delay = None
        self.delay_correlation_accumulator = None
        self.sample_used_for_delay_estimation = 0

    def set_full_ai_response(self, full_ai_response : np.array ):
        '''
        Setter for ai response
        :param full_ai_response:
        :return:
        '''
        self.full_ai_response = full_ai_response

    def reset(self):
        if self.delay_estimation_uncertainty_samples == 0:
            # delay estimation is disabled
            self.delay = 0
        else:
            # delay estimation is enabled - need to estimate the delay latter
            self.delay = None
        self.delay_correlation_accumulator = None
        self.sample_used_for_delay_estimation = 0

    @abstractmethod
    def filter_sample(self, x_new: float, d_sample: float, do_adapt: bool = True) :
        pass

    def estimate_delay(self, mic_batch : np.array, index_in_reference : int):
        '''
        Estimate the delay between reference and mic 
        perform correlation of each mic buffer respect the reference  , and accumulate the result in  self.delay_correlation_accumulator
        :param mic_batch: batch of mic signal
        :param index_in_reference: the matching index in reference buffer ( if delay was 0)  
        :return: 
        '''

        if len(self.full_ai_response) <  self.delay_estimation_uncertainty_samples:
            print('not enough samples for delay estimation in the recorded ai response')
            return
        
        if (self.sample_used_for_delay_estimation >   self.min_size_for_delay_estimation * 4) & (self.delay is not None):
            #print('stop estimation delay - enough is enough ')
            return
        
        if(len( self.full_ai_response) > index_in_reference+self.delay_estimation_uncertainty_samples ):

            # Estimate delay by correlation of the current mic batch respect the reference  
            signal = self.full_ai_response[index_in_reference:  index_in_reference+self.delay_estimation_uncertainty_samples].reshape(1, -1).astype(np.float32)
            template = mic_batch.reshape(1, -1).astype(np.float32)

            # Correlation normalized by energy of the signals  - approximation for normalized correlation  
            result = cv2.matchTemplate(signal, template, cv2.TM_CCOEFF).flatten() / (np.sqrt(np.mean((signal)**2)*np.mean((template)**2))*template.size)#

            # Update delay correlation buffer with the  correlation of the current mic batch 
            if self.delay_correlation_accumulator is None:
                self.delay_correlation_accumulator = result
            else:
                alpha =  mic_batch.size / self.min_size_for_delay_estimation
                # Update the correlation_accumulator with the new result (set alpha for exponential averaging over min_size_for_delay_estimation)
                self.delay_correlation_accumulator = self.delay_correlation_accumulator * (1-alpha) + result * alpha

            # Update number of samples used for delay estimation                 
            self.sample_used_for_delay_estimation += mic_batch.size

            if self.sample_used_for_delay_estimation > self.min_size_for_delay_estimation:
                # Enough samples were correlated =>  Check that correlation is good enough for setting the delay
                maxcorri = np.argmax( self.delay_correlation_accumulator)
                maxcorr =  self.delay_correlation_accumulator[maxcorri]
                # TODO - make a better criteria 
                if((maxcorr > 0.2 )  & (maxcorr / np.percentile(np.abs( self.delay_correlation_accumulator), 75) > 3)):
                    self.delay = maxcorri
                    print(f" max correlation between mic and ai response {np.max( self.delay_correlation_accumulator):.2f} @ {maxcorri} , used { self.sample_used_for_delay_estimation} samples ")

                # 
                # import pylab as plt
                # plt.figure()
                # plt.plot(template.flatten()/np.max(template))
                # plt.plot(signal.flatten() / np.max(signal))
                # plt.figure()
                # plt.plot(result.flatten(),label = 'TM_CCOEFF_NORMED')
                # plt.plot( self.delay_correlation_accumulator, label = 'delay_correlation_accumulator')
                # plt.legend()
                # plt.show()
                # # pass


    def process_batch(self, mic_batch: np.array, index_in_reference : int):
        '''
        process a batch of microphone samples
        :param mic_batch:
        :param index_in_reference: the matching index in the reference buffer
        :return:
        '''

        self.estimate_delay(mic_batch , index_in_reference)

        e_batch = mic_batch
        if self.delay is not None:
            d_batch = self.full_ai_response[self.delay+  self.additional_delay  + index_in_reference:self.delay+  self.additional_delay + index_in_reference+len(mic_batch)]
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
    def __init__(self, filter_length: int, full_ai_response : np.array ,  min_size_for_delay_estimation : int = 0,delay_estimation_uncertainty_samples : int = 0,
                 lmbd: float = 0.995, delta: float = 0.01):
        '''
        :param filter_length:
        '''
        super().__init__(filter_length, full_ai_response, min_size_for_delay_estimation, delay_estimation_uncertainty_samples)
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
    '''
    Simulation of delay
    :param inpath:
    :return:
    '''
    from adapt_utils import get_aligned_signals, display_and_save
    from copy import copy

    min_size_for_delay_estimation_sec = 0.25 # minimal number of time to use for valid delay estimation[sec]
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
    system_delay_sec = 0.1 # simulate this delay
    system_delay_samples = int(system_delay_sec * sr)

    if system_delay_samples > 0:
        ref = ref[:-system_delay_samples]
        mic = mic[system_delay_samples:]
    else:
        pass

    ####################################################

    # Define the filter
    adaptfilter = AdaptiveRLSFilter(64, ref,
                                    min_size_for_delay_estimation = int(min_size_for_delay_estimation_sec * sr) ,
                                    delay_estimation_uncertainty_samples = int(max_system_delay_sec * sr) )

    # Run the filter , feed it batch by batch of mic samples
    mic_filtered = np.zeros(len(mic))
    for b in range(0, len(mic), batch_size):
        e_batch = adaptfilter.process_batch(mic[b:b + batch_size], b)
        mic_filtered[b:b + batch_size] = e_batch

    display_and_save(mic, mic_filtered, ref, sr, inpath)

if __name__ == "__main__":
    run_sim('C:/Users/dadab/projects/AEC/data/rec1/2')
