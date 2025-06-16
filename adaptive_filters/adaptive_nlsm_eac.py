import numpy as np
from typing import Tuple
import cv2
import os
import soundfile as sf
import librosa
from adapt_utils import get_aligned_signals

class AdaptiveFilter:
    """
     adaptive nlms filter
    """

    def __init__(self, filter_length: int = 32, mu: float = 0.00):
        '''
        :param filter_length:  the filter length (samples
        :param mu: learning rate
        '''
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
        '''
        Reset adaptive filter state
        :return:
        '''

        self.h1.fill(0)
        self.x_buffer.fill(0)
        self.sigma_e = 0
        self.sigma_r = 0

    def update_buffer(self, x_new: float):
        '''
        Update input delay line
        :param x_new: input sample
        :return:
        '''

        self.x_buffer[1:] = self.x_buffer[:-1]
        self.x_buffer[0] = x_new

    def compute_output(self):
        '''
        Compute  filter output
        :return:  one sample of the output
        '''
        return np.dot(self.h1, self.x_buffer[:self.M1])

    def adapt(self, error: float):
        '''
        Adapt filter coefficients using NLMS algorithm
        :param error: the difference between the filter output and the reference
        :return:
        '''
        power = np.dot(self.x_buffer[:self.M1], self.x_buffer[:self.M1]) + self.regularization

        step = self.mu1 / power

        # Update the filter
        self.h1 += step * error * self.x_buffer[:self.M1]




    def filter_sample(self, x_new: float, d_sample: float, adapt: bool = True) -> Tuple:
        '''
        Process single sample
        :param x_new:   New input sample
        :param d_sample:    Desired signal sample
        :param adapt:  Whether to adapt coefficients
        :return:
        y : float
            Filter output
        e : float
            Error signal
        '''

        self.update_buffer(x_new)
        y = self.compute_output()
        e = d_sample - y


        if adapt:
            self.adapt(e)

        return y, e

    def process_signals(self, reference: np.array, mic: np.array):
        '''

        :param reference: the ai recorded signal
        :param mic:  mic signal
        :return:
        '''
        N = len(reference)
        echo_estimate = np.zeros(N)
        error_signal = np.zeros(N)

        for n in range(N):
            adapt = reference[n] != 0  # Do not adapt filter if the ai response is not active
            y, e = self.filter_sample(
                reference[n], mic[n], adapt=adapt
            )

            echo_estimate[n] = y
            error_signal[n] = e

        return echo_estimate, error_signal


def test_the_filter(inpath: str, output_name: str = None):

    mic, ref, sr = get_aligned_signals(inpath, 32)


    adapt_filter = AdaptiveFilter(filter_length=128, mu=1e-1)


    # Add talk to the mic
    # from scipy.signal import coherence
    # import pylab as plt
    # cr = []
    # for i in np.arange(1,len(mic),512):
    #     #f, Cxy = coherence(mic[first_ai_response:], ref[first_ai_response:], fs=sr, nperseg=512)
    #     f, Cxy = coherence(mic[i:i*512+1], ref[i:i*512+1], fs=sr, nperseg=512)
    #     cr.append(np.mean(Cxy))
    # plt.figure()
    # plt.plot(cr)


    first_ai_response = np.where(ref > 0)[0][0]
    # if True:
    #     talk_len = (len(mic)-first_ai_response)//4
    #     mic[first_ai_response+talk_len:first_ai_response+talk_len*2] = 0.0*mic[first_ai_response+talk_len:first_ai_response+talk_len*2] + mic[:len( mic[first_ai_response+talk_len:first_ai_response+talk_len*2])]*100
    #     # Windowed Coherence
    # from scipy.signal import coherence
    # import pylab as plt
    # cr = []
    # for i in np.arange(1,len(mic),512):
    #     #f, Cxy = coherence(mic[first_ai_response:], ref[first_ai_response:], fs=sr, nperseg=512)
    #     f, Cxy = coherence(mic[i:i*512], ref[i:i*512], fs=sr, nperseg=512)
    #     cr.append(np.mean(Cxy))
    # plt.figure()
    # plt.plot(cr)
    # plt.show()

    # run filter
    echo_estimate, mic_sig_filtered = adapt_filter.process_signals(
        ref, mic
    )


    ###################   Debug     ###################
    if output_name is not None:
        sf.write(os.path.join(inpath , output_name),
                      mic_sig_filtered,sr)
        sf.write(os.path.join(inpath , "used_mic.wav"),
                      mic,sr)


    # Calculate performance metrics - the attenuation of the ai response in the microphone
    first_ai_response = np.where(ref > 0)[0][0]

    initial_power = np.mean(mic[first_ai_response:] ** 2)
    final_power = np.mean(mic_sig_filtered[first_ai_response:] ** 2)

    print(
        f"Echo suppression:  ,  {10 * np.log10(initial_power / final_power):.2f} dB  (energy reduction by factor of  {(initial_power / final_power):.2f}) ")


if __name__ == "__main__":
    # Run the test
    test_the_filter('C:/Users/dadab/projects/AEC/data/rec1/2', 'mic_filtered_adaptive_nlms5.wav')