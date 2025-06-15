import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from scipy import signal
import os
from scipy.io import wavfile
import soundfile as sf
from copy import copy
import librosa

class AdaptiveFilter:
    """
     adaptive nlms filter
    """

    def __init__(self, filter_length: int =32,mu : float =0.00):
        '''
        :param filter_length:  the filter length (samples
        :param mu: learning rate
        '''
        self.M1 = filter_length
        self.mu1 = mu
        self.regularization = 1e-2
        # Initialize kernels (coefficients)
        self.h1 = np.zeros(self.M1)  # 1st order kernel (linear)

        # Input delay line
        self.x_buffer = np.zeros(self.M1)

        # Reset filter
        self.reset()

    def reset(self):
        """Reset adaptive filter state"""
        self.h1.fill(0)
        self.x_buffer.fill(0)

    def update_buffer(self, x_new : float):
        """Update input delay line"""
        self.x_buffer[1:] = self.x_buffer[:-1]
        self.x_buffer[0] = x_new

    def compute_output(self):
        '''
        Compute  filter output
        :return:  one sample of the output
        '''
        return np.dot(self.h1, self.x_buffer[:self.M1])


    def adapt(self, error : float):
        '''
        Adapt filter coefficients using NLMS algorithm
        :param error: the difference between the filter output and the reference
        :return:
        '''

        power = np.dot( self.x_buffer[:self.M1],  self.x_buffer[:self.M1]) / self.M1 + self.regularization
        step = self.mu1 / power
        # Update the filter
        self.h1 += step * error * self.x_buffer[:self.M1]


    def filter_sample(self, x_new: float, d_sample: float, adapt : bool=True)->Tuple:
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

    def process_signals(self, reference : np.array, mic : np.array):
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
                reference[n], mic[n], adapt= adapt
            )

            echo_estimate[n] = y
            error_signal[n] = e

        return echo_estimate, error_signal


# # Example usage and testing
# def test_filter(inpath):
#
#
#
#     mic, sr = sf.read(inpath + '/mic_output.wav')
#     #ref, sr = sf.read(inpath + '/resampled_and_normalized_ai.wav')
#     ref, sr = sf.read(inpath + '/resampled_ai.wav')
#     mic_sig_filtered = copy(mic)
#
#     first_ai_response = np.where(ref > 0)[0][0]
#     mic = mic[first_ai_response:]
#     ref = ref[first_ai_response:]
#     # add delay
#     gp = 32
#     mic = mic[:-gp]
#     ref = ref[gp:]
#
#     # scale_mic = mic.max()
#     # scale_ref = ref.max()
#
#     scale_mic = 1 #mic.max()
#     scale_ref = 1 #ref.max()
#
#
#     ref = ref  / scale_ref
#     mic = mic / scale_mic
#
#
#     filter = AdaptiveFilter(filter_length=128, mu=0.002)
#
#     # run filter
#     echo_estimate, error_signal = filter.process_signals(
#         ref, mic
#     )
#
#
#
#     mic_sig_filtered[first_ai_response:-gp] = error_signal * scale_mic
#     sf.write(os.path.join(inpath + '/mic_filtered_adaptive_nlms2.wav'),
#                   mic_sig_filtered,sr)
#
#
#     # Calculate performance metrics
#     initial_power = np.mean(mic ** 2)  # First second
#     final_power = np.mean(error_signal ** 2)  # Last second
#
#     print(f"Initial echo power: {10 * np.log10(initial_power):.2f} dB")
#     print(f"Final residual power: {10 * np.log10(final_power):.2f} dB")
#     #print(f"Echo suppression: {10 * np.log10(initial_power / final_power):.2f} dB")
#     print(f"Echo suppression: {(initial_power / final_power):.2f}")

# # Example usage and testing
def test_the_filter(inpath : str):
    # Read mic signal
    mic, sr = sf.read(inpath + '/mic_output.wav')
    # Read ai signal
    ref, sr = sf.read(inpath + '/resampled_ai.wav')



    # Add delay
    gp = 32
    mic = mic[:-gp]
    ref = ref[gp:]

    # ref = ref  / scale_ref
    # mic = mic / scale_mic


    filter = AdaptiveFilter(filter_length=128, mu=0.002)

    # run filter
    echo_estimate, error_signal = filter.process_signals(
        ref, mic
    )



    mic_sig_filtered = error_signal# * scale_mic
    sf.write(os.path.join(inpath + '/mic_filtered_adaptive_nlms2.wav'),
                  mic_sig_filtered,sr)


    # Calculate performance metrics - the attenuation of the ai response in the microphone
    first_ai_response = np.where(ref > 0)[0][0]

    initial_power = np.mean(mic[first_ai_response:] ** 2)
    final_power = np.mean(mic_sig_filtered[first_ai_response:]  ** 2)

    print(f"Echo suppression: {(initial_power / final_power):.2f} ,  {10 * np.log10(initial_power / final_power):.2f} dB ")

if __name__ == "__main__":
    # Run the test
    test_the_filter('C:/Users/dadab/projects/AEC/data/rec1/2')

