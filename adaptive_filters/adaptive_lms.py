import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from scipy.io import wavfile
import soundfile as sf

class NLMSAdaptiveFilter:
    def __init__(self, filter_length=16, step_size=0.5, regularization=1e-6):
        """
        Normalized LMS Adaptive Filter

        Args:
            filter_length: Length of the adaptive filter
            step_size: Learning rate (0 < mu < 2)
            regularization: Small value to prevent division by zero
        """
        self.filter_length = filter_length
        self.step_size = step_size
        self.regularization = regularization
        self.weights = np.zeros(filter_length)
        self.reference_buffer = np.zeros(filter_length)

    def process_sample(self, reference_sample, mixed_sample):
        """Process a single sample using NLMS"""
        # Update reference buffer
        self.reference_buffer = np.roll(self.reference_buffer, 1)
        self.reference_buffer[0] = reference_sample

        # Estimate interference
        interference_estimate = np.dot(self.weights, self.reference_buffer)

        # Calculate error (cleaned signal)
        error = mixed_sample - interference_estimate

        # Normalized step size
        power = np.dot(self.reference_buffer, self.reference_buffer) + self.regularization
        normalized_step = self.step_size / power

        # Update weights
        self.weights += normalized_step * error * self.reference_buffer

        return error

    def process_signals(self, reference_signal, mixed_signal):
        """Process entire signals"""
        min_len = min(len(reference_signal), len(mixed_signal))
        reference_signal = reference_signal[:min_len]
        mixed_signal = mixed_signal[:min_len]

        cleaned_signal = np.zeros(min_len)

        for i in range(min_len):
            cleaned_signal[i] = self.process_sample(reference_signal[i], mixed_signal[i])

        return cleaned_signal


class VolterraFilter:
    """
    Volterra filter implementation for nonlinear system identification
    Supports 1st and 2nd order kernels
    """

    def __init__(self, memory_length_1=32, memory_length_2=32, mu1=0.001, mu2=0.001):
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
        self.M1 = memory_length_1
        self.M2 = memory_length_2
        self.mu1 = mu1
        self.mu2 = mu2

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
        """Compute Volterra filter output"""
        # 1st order (linear) output
        y1 = np.dot(self.h1, self.x_buffer[:self.M1])

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
        self.h1 += self.mu1 * error * self.x_buffer[:self.M1]

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


class EchoCanceller:
    """
    Echo cancellation system using Volterra filters
    """

    def __init__(self, linear_length=128, nonlinear_length=32,
                 mu_linear=0.01, mu_nonlinear=0.001):
        """
        Initialize echo canceller

        Parameters:
        -----------
        linear_length : int
            Length of linear echo path model
        nonlinear_length : int
            Length of nonlinear components
        mu_linear : float
            Adaptation step size for linear part
        mu_nonlinear : float
            Adaptation step size for nonlinear part
        """
        self.filter = VolterraFilter(
            memory_length_1=linear_length,
            memory_length_2=nonlinear_length,
            mu1=mu_linear,
            mu2=mu_nonlinear
        )

        # Performance monitoring
        self.error_power = []
        self.echo_power = []
        self.echo_return_loss = []

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
            y, e = self.filter.filter_sample(
                far_end[n], near_end[n], adapt=adapt
            )

            echo_estimate[n] = y
            error_signal[n] = e

        return echo_estimate, error_signal




# Example usage and testing
def test_filter(inpath):

    import soundfile as sf
    from copy import copy
    import librosa

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

    scale_mic = mic.max()

    far_end = ref  / ref.max()
    near_end = mic / scale_mic
    fs = sr


    canceller = EchoCanceller(
        linear_length=128,
        nonlinear_length=32,
        mu_linear=0.05,
        mu_nonlinear=0.005
    )

    print("Running Volterra echo cancellation...")

    # Process signals
    echo_estimate, error_signal = canceller.process_signals(
        far_end, near_end, adapt=True
    )



    mic_sig_filtered[first_ai_response:-gp] = error_signal * scale_mic
    sf.write(os.path.join(inpath + '/mic_filtered_adaptive_voltera.wav'),
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

