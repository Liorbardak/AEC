import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
import cv2
import time
from copy import copy
class AdaptiveFilter(ABC):
    """
     adaptive filter base
    """
    def __init__(self, filter_length: int, min_size_for_delay_estimation: int = 0,
                 delay_estimation_uncertainty_samples: int = 0 ,
                 reduced_output_factor: float = 0.5,
                 additional_delay :int = 64,
                 sample_rate: int = 16000):
        '''

        :param filter_length: adaptive filter length
        :param min_size_for_delay_estimation:minimal number of samples to match for delay estimation [samples]
        :param delay_estimation_uncertainty_samples: delay uncertainty[samples] , if = 0 - don't estimate delay
        :param reduced_output_factor:  output attenuation factor at the first samples , when filter is not active   , put 1.0 to disable
        :param additional_delay - additional delay [samples] for AI response signal before filtering - allow the casual filter to work better
        :param sample_rate: sample rate [Hz] - for debug/logging
        '''

        self.filter_length = filter_length

        self.min_size_for_delay_estimation = min_size_for_delay_estimation

        self.delay_estimation_uncertainty_samples = delay_estimation_uncertainty_samples

        self.additional_delay = additional_delay  # add a bit more delay to the ai response , improve the attenuation of the casual filter

        self.sample_rate = sample_rate # debug

        self.reduced_output_length_samples = self.delay_estimation_uncertainty_samples +  self.min_size_for_delay_estimation

        self.reduced_output_factor = reduced_output_factor

        self.reduced_output_shift_period = 8000 # shifting from reduced output to "normal" output


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
        perform correlation of mic buffer respect the reference
        :param mic_batch: batch of mic signal
        :return:
        '''

        # Delay estimation not needed
        if (self.delay_estimation_uncertainty_samples == 0) | (self.min_size_for_delay_estimation == 0):
            return

        # Delay was estimated already - no need to re-estimate
        if (self.delay is not None):
            return

        # Add the mic samples to the local mic buffer
        self.mic_buffer = np.concatenate((self.mic_buffer, mic_batch))

        # look for the actual start of the AI response - omit the silence
        valid_aisig = np.where(self.ai_response_buffer > 0.01)[0]
        if len(valid_aisig) == 0:
            print('no information in ai response ')
            return
        # The first valid AI sample
        ai_offset = valid_aisig[0]

        if (len(self.mic_buffer) < self.delay_estimation_uncertainty_samples + self.min_size_for_delay_estimation+ ai_offset):
            print(f"can not calculate delay yet , need more mic samples ,   mic length: {self.mic_sample_index} <   {self.delay_estimation_uncertainty_samples + self.min_size_for_delay_estimation + ai_offset} ")
            return

        if (len(self.ai_response_buffer) < len(self.mic_buffer) - self.delay_estimation_uncertainty_samples):
            print(f"can not calculate delay , need more ai samples,   mic length: {self.mic_sample_index} ai response len:  {len(self.ai_response_buffer) } , threshold :  {len(self.mic_buffer) - self.delay_estimation_uncertainty_samples}")
            return

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
        # Find second-best correlation peak
        nonmax_gap = 1000
        second_peak = 0
        if maxcorri > nonmax_gap:
            second_peak = np.max(result[:maxcorri - nonmax_gap])
        if maxcorri < len(result)-nonmax_gap-2:
            second_peak = np.max([second_peak, np.max(result[maxcorri+nonmax_gap:])])

         # delay estimation success criteria TODO - make a better criteria
        if ((maxcorr > 0.2) & (maxcorr / second_peak > 1.4)):
            self.delay = maxcorri
            print(f" delay found: {self.delay}[samples] , correlation: {maxcorr:.2f}   second peak: {second_peak:.2f} mic length: {self.mic_sample_index} ai response len:  {len(self.ai_response_buffer)} ")

            # Debug display
            print(len(template.flatten()))
            import pylab as plt
            plt.subplot(2, 1, 1)
            plt.plot(template.flatten()/np.max(template), label='template')
            plt.plot(signal.flatten() / np.max(signal), label='signal')
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(result.flatten(),label = 'TM_CCOEFF_NORMED')
            plt.legend()
            plt.show()
        else:
            print(f'delay estimation failed, best delay : {maxcorri} correlation  {maxcorr:.2f}   second peak {second_peak:.2f} mic length: {self.mic_sample_index} ai response len:  {len(self.ai_response_buffer)} ')

    def process_batch(self, mic_batch: np.array):
        '''
        process a batch of microphone samples
        :param mic_batch:
        :param index_in_reference: the matching index in the reference buffer
        :return:
        '''

        # Estimate coarse delay
        self.estimate_delay(mic_batch)

        e_batch = copy(mic_batch) # default - output = input  , any without filtering
        if self.delay is not None:


            # Delay was estimated => can filer
            # Get the matching AI samples
            d_batch = self.ai_response_buffer[
                      -self.delay + self.additional_delay + self.mic_sample_index :-self.delay + self.additional_delay +      self.mic_sample_index  + len(
                          mic_batch)]
            if (d_batch.size == mic_batch.size):
                # Apply the adaptive filter             
                start = time.perf_counter()
                
                e_batch = self.filter_batch(d_batch, mic_batch)
                
                end = time.perf_counter()
             

                # Store statistics
                if  self.stat['filtering_time_per_sample'] is None:
                    self.stat['filtering_time_per_sample'] = (end - start) / len(mic_batch)
                else:
                    self.stat['filtering_time_per_sample'] = self.stat['filtering_time_per_sample'] * 0.95 + (end - start) / len(mic_batch) * 0.05
                self.stat['mic_energy_before'] += np.sum(mic_batch**2)
                self.stat['mic_energy_after'] +=  np.sum(e_batch ** 2)
                self.stat['n_samples_filtered'] += len(mic_batch)
                self.stat['print_cnt'] +=  len(mic_batch)

        # Attenuate the first output  samples  , when the filter is not active yet
        for i in np.arange(len(e_batch)):
            ind = i + self.mic_sample_index
            if ind < self.reduced_output_length_samples:
                e_batch[i] = e_batch[i] * self.reduced_output_factor
            elif ind < self.reduced_output_length_samples  + self.reduced_output_shift_period:
                # smoothly increase the amplitude back to normal
                alpha = (ind - self.reduced_output_length_samples) /  self.reduced_output_shift_period
                # ind = self.reduced_output_length_samples => alpha = 0 => factor = self.reduced_output_factor
                # ind = self.reduced_output_length_samples+  self.reduced_output_shift_period => alpha = 1 => factor = 1.0
                e_batch[i] = e_batch[i] * (alpha * 1.0 + (1 - alpha) * self.reduced_output_factor)
            else:
                pass

        self.mic_sample_index += len(mic_batch)

        # Debug printing
        if(self.stat['print_cnt']  >  self.sample_rate * 0.1):
            self.stat['print_cnt'] = 0
            print(f"filter is running , time [%] {100 * self.stat['filtering_time_per_sample']* self.sample_rate :.1f}  , energy reduction by a factor of    {self.stat['mic_energy_before'] /    self.stat['mic_energy_after'] :.2f}")

        return e_batch

    def filter_batch(self, mic_batch: np.array, ai_batch: np.array):
        '''
        process a batch of samples
        '''
        e_batch = np.zeros_like(mic_batch)
        for i in np.arange(len(mic_batch)):
            e_batch[i] = self.filter_sample(mic_batch[i], ai_batch[i], do_adapt= ai_batch[i] != 0)
        return e_batch

    def  add_ai_response_batch(self, ai_batch: np.array):
        '''
        add  a batch of AI response samples to the internal buffer
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
        :param lmbd - exponential window
        :param delta -  covariance init
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
                 mu: float = 1e-1,
                 regularization: float = 1e-4):
        '''
        :param filter_length:
        :param mu - learning rate (0< mu <2 )
        :param regularization - power regularization
        '''
        super().__init__(filter_length, min_size_for_delay_estimation,
                         delay_estimation_uncertainty_samples)
        self.mu = mu

        self.filter_length = filter_length
        self.mu = mu
        self.regularization = regularization  # power estimation regularization factor

        # Initialize kernels (coefficients)
        self.w = np.zeros(self.filter_length)

        # Input delay line
        self.x_buffer = np.zeros(self.filter_length)

        # Reset filter
        self.reset()


    def reset(self):
        super().reset()
        self.w.fill(0)
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
        y =  np.dot(self.w, self.x_buffer[:self.filter_length])

        e = d_sample - y

        if do_adapt:
            # Adapt the filter
            power = np.dot(self.x_buffer[:self.filter_length], self.x_buffer[:self.filter_length]) + self.regularization
            step =self.mu / power

            self.w += step * e * self.x_buffer[:self.filter_length]

        return e
