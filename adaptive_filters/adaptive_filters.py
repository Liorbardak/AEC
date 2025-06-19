import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
import cv2
import time

class AdaptiveFilter(ABC):
    """
     adaptive filter base
    """

    def __init__(self, filter_length: int, min_size_for_delay_estimation: int = 0,
                 delay_estimation_uncertainty_samples: int = 0 , sample_rate: int = 16000):
        '''

        :param filter_length: adaptive filter length
        :param full_ai_response: the reference  ai response signal
        :param min_size_for_delay_estimation:minimal number of samples to match for delay estimation [samples]
        :param delay_estimation_uncertainty_samples: delay uncertainty[samples] , if = 0 - dont estimate delay
        '''

        self.filter_length = filter_length

        self.min_size_for_delay_estimation = min_size_for_delay_estimation

        self.delay_estimation_uncertainty_samples = delay_estimation_uncertainty_samples

        self.additional_delay = 48  # add a bit more delay to the ai response , improve the attenuation of the casual filter

        self.sample_rate = sample_rate # ge
        
        
        if (self.delay_estimation_uncertainty_samples == 0) | (self.min_size_for_delay_estimation == 0):
            # delay estimation is disabled
            self.delay = 0
        else:
            # delay estimation is enabled - need to estimate the delay latter
            self.delay = None

        self.mic_buffer  = np.array([])
        self.ai_response_buffer = np.array([])
        self.mic_sample_index = 0

        self.stat = {'mic_energy_before': 0,'mic_energy_after': 0 ,
                     'n_samples_filtered' : 0,
                     'print_cnt': 0,
                     'filtering_time_per_sample'  : None}


    def reset(self):
        if self.delay_estimation_uncertainty_samples == 0:
            # delay estimation is disabled
            self.delay = 0
        else:
            # delay estimation is enabled - need to estimate the delay latter
            self.delay = None

        self.mic_buffer  = np.array([])
        self.ai_response_buffer = np.array([])
        self.mic_sample_index = 0

    @abstractmethod
    def filter_sample(self, x_new: float, d_sample: float, do_adapt: bool = True):
        pass

    def estimate_delay(self, mic_batch: np.array):
        '''
        Estimate the delay between reference and mic
        perform correlation of each mic buffer respect the reference  , and accumulate the result in  self.delay_correlation_accumulator
        :param mic_batch: batch of mic signal
        :return:
        '''

        # No delay estimation need
        if (self.delay_estimation_uncertainty_samples == 0) | (self.min_size_for_delay_estimation == 0):
            return

        # delay was estimated allready
        if (self.delay is not None):
            return

        # add the samples to the local mic buffer
        self.mic_buffer = np.concatenate((self.mic_buffer, mic_batch))

        # look for the actual start of the ai response - omit the silence
        valid_aisig = np.where(self.ai_response_buffer > 0.01)[0]
        if len(valid_aisig) == 0:
            print('no information in ai response ')
            return
        # The first valid ai sample
        ai_offset = valid_aisig[0]

        if (self.mic_sample_index  < self.delay_estimation_uncertainty_samples + self.min_size_for_delay_estimation+ ai_offset):
            print(f"can not calculate delay yet , need more mic samples {self.mic_sample_index} <   {self.delay_estimation_uncertainty_samples + self.min_size_for_delay_estimation + ai_offset} ")
            return
        if (len(self.ai_response_buffer) < self.mic_sample_index - self.delay_estimation_uncertainty_samples - self.min_size_for_delay_estimation + ai_offset):
            print('can not calculate delay , need more ai samples ')
            return

        print('estimating delay')

        # Estimate delay by correlation of the current mic batch respect the reference

        template = self.ai_response_buffer[ai_offset:len(self.mic_buffer) - self.delay_estimation_uncertainty_samples].reshape(1,
                                                                                                             -1).astype(
            np.float32)

        signal =     self.mic_buffer[ai_offset:].reshape(1, -1).astype(np.float32)


        # Template matching by NCC
        result = cv2.matchTemplate(signal, template, cv2.TM_CCOEFF_NORMED).flatten()

        # Enough samples were correlated =>  Check that correlation is good enough for setting the delay
        maxcorri = np.argmax(result)
        maxcorr = result[maxcorri]
        # TODO - make a better criteria
        if ((maxcorr > 0.2) & (maxcorr / np.percentile(np.abs(result), 75) > 3)):
            self.delay = maxcorri
            print(f" delay found {self.delay}, correlation  {np.max(result):.2f}")


            #import pylab as plt
            #plt.subplot(2, 1, 1)
            #plt.plot(template.flatten()/np.max(template), label='template')
            #plt.plot(signal.flatten() / np.max(signal), label='signal')
            #plt.legend()
            #plt.subplot(2, 1, 2)
            #plt.plot(result.flatten(),label = 'TM_CCOEFF_NORMED')
            #plt.legend()
            #plt.show()


    def process_batch(self, mic_batch: np.array):
        '''
        process a batch of microphone samples
        :param mic_batch:
        :param index_in_reference: the matching index in the reference buffer
        :return:
        '''

        # Estimate coarse delay
        self.estimate_delay(mic_batch)


        e_batch = mic_batch
        if self.delay is not None:
            # Delay was estimated - can filer
            # Get the matching ai samples
            d_batch = self.ai_response_buffer[
                      -self.delay + self.additional_delay + self.mic_sample_index :-self.delay + self.additional_delay +      self.mic_sample_index  + len(
                          mic_batch)]
            if (d_batch.size == mic_batch.size):
                # Apply the adaptive filter
                #start = time.time()
                start = time.perf_counter()
                e_batch = self.filter_batch(d_batch, mic_batch)
                end = time.perf_counter()
                #end = time.time()


                # Store statistics
                if  self.stat['filtering_time_per_sample'] is None:
                    self.stat['filtering_time_per_sample'] = (end - start) / len(mic_batch)
                else:
                    self.stat['filtering_time_per_sample'] = self.stat['filtering_time_per_sample'] * 0.95 + (end - start) / len(mic_batch) * 0.05
                self.stat['mic_energy_before'] += np.sum(mic_batch**2)
                self.stat['mic_energy_after'] +=  np.sum(e_batch ** 2)
                self.stat['n_samples_filtered'] += len(mic_batch)
                self.stat['print_cnt'] +=  len(mic_batch)


        self.mic_sample_index += len(mic_batch)

        if(self.stat['print_cnt']  > 8000):
            self.stat['print_cnt'] = 0
            print(f"time [%] {100 * self.stat['filtering_time_per_sample']* self.sample_rate :.1f}  , filter energy reduction    {self.stat['mic_energy_before'] /    self.stat['mic_energy_after'] :.2f}")

        return e_batch

    def filter_batch(self, mic_batch: np.array, d_batch: np.array):
        '''
        process a batch of samples
        '''
        e_batch = np.zeros_like(mic_batch)
        for i in np.arange(len(mic_batch)):
            e_batch[i] = self.filter_sample(mic_batch[i], d_batch[i], do_adapt=mic_batch[i] != 0)

        return e_batch

    def  add_ai_response_batch(self, ai_batch: np.array):
        '''
        add  a batch of ai response samples to the internal buffer
        '''
        self.ai_response_buffer = np.concatenate((self.ai_response_buffer, ai_batch))


class AdaptiveRLSFilter(AdaptiveFilter):
    """
    rls adaptive filter
    """

    def __init__(self, filter_length: int, min_size_for_delay_estimation: int = 0,
                 delay_estimation_uncertainty_samples: int = 0,
                 lmbd: float = 0.995, delta: float = 0.01):
        '''
        :param filter_length:
        '''
        super().__init__(filter_length, min_size_for_delay_estimation,
                         delay_estimation_uncertainty_samples)
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

class AdaptiveNLMSFilter(AdaptiveFilter):
    """
    nlms adaptive filter
    """

    def __init__(self, filter_length: int, min_size_for_delay_estimation: int = 0,
                 delay_estimation_uncertainty_samples: int = 0,
                 mu: float = 1e-1):
        '''
        :param filter_length:
        '''
        super().__init__(filter_length, min_size_for_delay_estimation,
                         delay_estimation_uncertainty_samples)
        self.mu = mu

        self.M1 = filter_length
        self.mu1 = mu
        self.regularization = 1e-4  # power estimation regularization factor
        # Initialize kernels (coefficients)
        self.h1 = np.zeros(self.M1)  # 1st order kernel (linear)

        # Input delay line
        self.x_buffer = np.zeros(self.M1)

        # Reset filter
        self.reset()


    def reset(self):
        super().reset()

        self.h1.fill(0)
        self.x_buffer.fill(0)


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
        # Update buffer
        self.x_buffer[1:] = self.x_buffer[:-1]
        self.x_buffer[0] = x_new

        # Get output
        y =  np.dot(self.h1, self.x_buffer[:self.M1])

        e = d_sample - y

        if do_adapt:
            # Adapt the filter
            power = np.dot(self.x_buffer[:self.M1], self.x_buffer[:self.M1]) + self.regularization
            step = self.mu1 / power

            self.h1 += step * e * self.x_buffer[:self.M1]

        return e
