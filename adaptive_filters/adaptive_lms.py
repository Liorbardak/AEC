import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from scipy.io import wavfile
import soundfile as sf
from copy import copy
import librosa

class AdaptiveFilter:
    """
    Nlms / lms adaptive filter
    Also support Volterra filter implementation for nonlinear system identification with 1st and 2nd order kernels
    """

    def __init__(self, filter_length=32,mu=0.001,  nonlinear_filter_length=0 , nonlinear_mu=0 ,
                 use_nlms = True):
        """
        Initialize Volterra filter

        Parameters:
        -----------
        memory_length_1 : int
            Memory length for 1st order kernel (linear part)
        memory_length_2 : int
            Memory length for 2nd order kernel (nonlinear part)
        mu1 : float
            Step size for 1st order adaptation
        mu2 : float
            Step size for 2nd order adaptation
        """
        self.M1 = filter_length
        self.M2 = nonlinear_filter_length
        self.mu1 = mu
        self.mu2 = nonlinear_mu
        self.regularization = 1e-2
        self.use_nlms =  use_nlms
        self.use_voltera = nonlinear_filter_length > 0
        # Initialize kernels (coefficients)
        self.h1 = np.zeros(self.M1)  # 1st order kernel (linear)
        self.h2 = np.zeros((self.M2, self.M2))  # 2nd order kernel (quadratic)

        # Input delay line
        self.x_buffer = np.zeros(max(self.M1, self.M2))

        # For adaptation
        self.reset_adaptation()

    def reset_adaptation(self):
        """Reset adaptive filter state"""
        self.h1.fill(0)
        self.h2.fill(0)
        self.x_buffer.fill(0)

    def update_buffer(self, x_new):
        """Update input delay line"""
        self.x_buffer[1:] = self.x_buffer[:-1]
        self.x_buffer[0] = x_new

    def compute_output(self):
        """Compute  filter output"""
        # 1st order (linear) output
        y1 = np.dot(self.h1, self.x_buffer[:self.M1])
        if  self.use_voltera:
            # 2nd order (quadratic) output
            y2 = 0
            for i in range(self.M2):
                for j in range(i, self.M2):  # Symmetric kernel
                    if i == j:
                        y2 += self.h2[i, j] * self.x_buffer[i] * self.x_buffer[j]
                    else:
                        y2 += 2 * self.h2[i, j] * self.x_buffer[i] * self.x_buffer[j]
            return y1 + y2

        return y1

    def adapt(self, error):
        """
        Adapt filter coefficients using LMS algorithm

        Parameters:
        -----------
        error : float
            Error signal for adaptation
        """
        # Update 1st order kernel
        # Normalized step size
        if self.use_nlms:
            # NLMS
            power = np.dot( self.x_buffer[:self.M1],  self.x_buffer[:self.M1]) / self.M1 + self.regularization
            step = self.mu1 / power
        else:
            # LMS
            step=  self.mu1


        self.h1 += step * error * self.x_buffer[:self.M1]

        if self.use_voltera:
            # Update 2nd order kernel
            for i in range(self.M2):
                for j in range(i, self.M2):
                    gradient = self.x_buffer[i] * self.x_buffer[j]
                    if i != j:
                        gradient *= 2  # Account for symmetry
                    self.h2[i, j] += self.mu2 * error * gradient

    def filter_sample(self, x_new, d_sample, adapt=True):
        """
        Process single sample

        Parameters:
        -----------
        x_new : float
            New input sample
        d_sample : float
            Desired signal sample
        adapt : bool
            Whether to adapt coefficients

        Returns:
        --------
        y : float
            Filter output
        e : float
            Error signal
        """
        self.update_buffer(x_new)
        y = self.compute_output()
        e = d_sample - y

        if adapt:
            self.adapt(e)

        return y, e
    def process_signals(self, far_end, near_end, adapt=True):
            """
            Process audio signals for echo cancellation

            Parameters:
            -----------
            far_end : array_like
                Far-end signal (reference)
            near_end : array_like
                Near-end signal with echo
            adapt : bool
                Enable adaptation

            Returns:
            --------
            echo_estimate : ndarray
                Estimated echo signal
            error_signal : ndarray
                Echo-cancelled signal
            """
            N = len(far_end)
            echo_estimate = np.zeros(N)
            error_signal = np.zeros(N)

            for n in range(N):
                y, e = self.filter_sample(
                    far_end[n], near_end[n], adapt=adapt
                )

                echo_estimate[n] = y
                error_signal[n] = e

            return echo_estimate, error_signal


# Example usage and testing
def test_filter(inpath):



    mic, sr = sf.read(inpath + '/mic_output.wav')
    ref, sr = sf.read(inpath + '/resampled_and_normalized_ai.wav')
    mic_sig_filtered = copy(mic)

    first_ai_response = np.where(ref > 0)[0][0]
    mic = mic[first_ai_response:]
    ref = ref[first_ai_response:]
    # add delay
    gp = 10
    mic = mic[:-gp]
    ref = ref[gp:]

    # scale_mic = mic.max()
    # scale_ref = ref.max()

    scale_mic = mic.max()
    scale_ref = ref.max()


    far_end = ref  /scale_ref
    near_end = mic / scale_mic
    fs = sr

    filter = AdaptiveFilter(filter_length=128, mu=0.002 )
   # filter = AdaptiveFilter(filter_length=128, mu=0.002 , nonlinear_filter_length=32 , nonlinear_mu=0.0005 )

    # run filter
    echo_estimate, error_signal = filter.process_signals(
        far_end, near_end, adapt=True
    )



    mic_sig_filtered[first_ai_response:-gp] = error_signal * scale_mic
    sf.write(os.path.join(inpath + '/mic_filtered_adaptive_nlms.wav'),
                  mic_sig_filtered,sr)


    # Calculate performance metrics
    initial_power = np.mean(near_end ** 2)  # First second
    final_power = np.mean(error_signal ** 2)  # Last second

    print(f"Initial echo power: {10 * np.log10(initial_power):.2f} dB")
    print(f"Final residual power: {10 * np.log10(final_power):.2f} dB")
    #print(f"Echo suppression: {10 * np.log10(initial_power / final_power):.2f} dB")
    print(f"Echo suppression: {(initial_power / final_power):.2f}")




if __name__ == "__main__":
    # Run the test
    results = test_filter('C:/Users/dadab/projects/AEC/data/rec1/2')

    # Optional: Save results to audio files
    # sf.write('far_end.wav', results['far_end'], 8000)
    # sf.write('near_end_with_echo.wav', results['near_end'], 8000)
    # sf.write('echo_cancelled.wav', results['error_signal'], 8000)

