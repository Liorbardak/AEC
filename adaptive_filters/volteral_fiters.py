import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
from scipy.io import wavfile
import soundfile as sf


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

        # 2nd order (quadratic) output
        y2 = 0
        for i in range(self.M2):
            for j in range(i, self.M2):  # Symmetric kernel
                if i == j:
                    y2 += self.h2[i, j] * self.x_buffer[i] * self.x_buffer[j]
                else:
                    y2 += 2 * self.h2[i, j] * self.x_buffer[i] * self.x_buffer[j]

        return y1 + y2

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


class VolterraEchoCanceller:
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
        self.volterra = VolterraFilter(
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
            y, e = self.volterra.filter_sample(
                far_end[n], near_end[n], adapt=adapt
            )

            echo_estimate[n] = y
            error_signal[n] = e

            # Monitor performance every 1000 samples
            if n % 1000 == 0 and n > 0:
                self._update_performance_metrics(
                    near_end[max(0, n - 1000):n],
                    error_signal[max(0, n - 1000):n]
                )

        return echo_estimate, error_signal

    def _update_performance_metrics(self, echo_signal, error_signal):
        """Update performance monitoring metrics"""
        echo_pow = np.mean(echo_signal ** 2)
        error_pow = np.mean(error_signal ** 2)

        self.echo_power.append(echo_pow)
        self.error_power.append(error_pow)

        # Echo Return Loss Enhancement (ERLE)
        if error_pow > 1e-10:
            erle = 10 * np.log10(echo_pow / error_pow)
        else:
            erle = 60  # Maximum ERLE

        self.echo_return_loss.append(erle)

    def get_coefficients(self):
        """Get current filter coefficients"""
        return {
            'linear_coeffs': self.volterra.h1.copy(),
            'nonlinear_coeffs': self.volterra.h2.copy()
        }

    def reset(self):
        """Reset the echo canceller"""
        self.volterra.reset_adaptation()
        self.error_power.clear()
        self.echo_power.clear()
        self.echo_return_loss.clear()


def create_nonlinear_echo_path(far_end, linear_coeffs, nonlinear_gain=0.1, delay=0):
    """
    Create synthetic nonlinear echo path for testing

    Parameters:
    -----------
    far_end : array_like
        Far-end signal
    linear_coeffs : array_like
        Linear echo path coefficients
    nonlinear_gain : float
        Strength of nonlinear distortion
    delay : int
        Echo delay in samples

    Returns:
    --------
    echo : ndarray
        Nonlinear echo signal
    """
    # Linear echo component
    echo_linear = signal.lfilter(linear_coeffs, 1, far_end)

    # Add delay if specified
    if delay > 0:
        echo_linear = np.concatenate([np.zeros(delay), echo_linear[:-delay]])

    # Nonlinear distortion (cubic nonlinearity)
    echo_nonlinear = nonlinear_gain * echo_linear ** 3

    # Combine linear and nonlinear components
    echo = echo_linear + echo_nonlinear

    return echo


def analyze_volterra_performance(canceller, far_end, near_end, echo_estimate, error_signal):
    """
    Analyze and plot echo cancellation performance
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # Time domain signals
    t = np.arange(len(far_end)) / 8000  # Assume 8kHz sampling

    axes[0, 0].plot(t, far_end, label='Far-end', alpha=0.7)
    axes[0, 0].plot(t, near_end, label='Near-end (with echo)', alpha=0.7)
    axes[0, 0].plot(t, error_signal, label='Echo cancelled', alpha=0.8)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Time Domain Signals')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Echo vs estimate
    axes[0, 1].plot(t, near_end - far_end, label='True echo', alpha=0.7)
    axes[0, 1].plot(t, echo_estimate, label='Echo estimate', alpha=0.7)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].set_title('Echo Estimation')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Linear coefficients
    coeffs = canceller.get_coefficients()
    axes[1, 0].plot(coeffs['linear_coeffs'])
    axes[1, 0].set_xlabel('Tap Index')
    axes[1, 0].set_ylabel('Coefficient Value')
    axes[1, 0].set_title('Linear Kernel (1st Order)')
    axes[1, 0].grid(True)

    # Nonlinear coefficients (2D heatmap)
    im = axes[1, 1].imshow(coeffs['nonlinear_coeffs'], aspect='auto', cmap='RdBu_r')
    axes[1, 1].set_xlabel('Tap Index j')
    axes[1, 1].set_ylabel('Tap Index i')
    axes[1, 1].set_title('Nonlinear Kernel (2nd Order)')
    plt.colorbar(im, ax=axes[1, 1])

    # Performance metrics over time
    if canceller.echo_return_loss:
        time_blocks = np.arange(len(canceller.echo_return_loss)) * 1000 / 8000
        axes[2, 0].plot(time_blocks, canceller.echo_return_loss)
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('ERLE (dB)')
        axes[2, 0].set_title('Echo Return Loss Enhancement')
        axes[2, 0].grid(True)

    # Power spectral densities
    f, Pxx_echo = signal.welch(near_end, fs=8000, nperseg=512)
    f, Pxx_cancelled = signal.welch(error_signal, fs=8000, nperseg=512)

    axes[2, 1].semilogy(f, Pxx_echo, label='With echo', alpha=0.7)
    axes[2, 1].semilogy(f, Pxx_cancelled, label='Echo cancelled', alpha=0.7)
    axes[2, 1].set_xlabel('Frequency (Hz)')
    axes[2, 1].set_ylabel('PSD')
    axes[2, 1].set_title('Power Spectral Density')
    axes[2, 1].legend()
    axes[2, 1].grid(True)

    plt.tight_layout()
    plt.show()


# Example usage and testing
def test_volterra_echo_cancellation(inpath):
    """Test the Volterra echo cancellation system"""

    # # Generate test signals
    # fs = 8000  # 8 kHz sampling rate
    # duration = 10  # seconds
    # t = np.arange(0, duration, 1 / fs)
    #
    # # Far-end speech signal (multi-tone for testing)
    # far_end = (0.5 * np.sin(2 * np.pi * 300 * t) +
    #            0.3 * np.sin(2 * np.pi * 800 * t) +
    #            0.2 * np.sin(2 * np.pi * 1500 * t) +
    #            0.1 * np.random.randn(len(t)))  # Add some noise
    #
    # # Create synthetic nonlinear echo path
    # # Linear echo path (room impulse response approximation)
    # echo_delay = int(0.05 * fs)  # 50ms delay
    # linear_path = np.exp(-np.arange(64) * 0.1) * np.random.randn(64) * 0.3
    # # Generate echo with nonlinear distortion
    # echo = create_nonlinear_echo_path(
    #     far_end, linear_path, nonlinear_gain=0.05, delay=echo_delay
    # )
    #
    # # Near-end signal (echo + local speech)
    # local_speech = 0.1 * np.sin(2 * np.pi * 500 * t + np.pi / 4)  # Local speaker
    # near_end = echo + local_speech
    #

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

    # Initialize Volterra echo canceller
    # canceller = VolterraEchoCanceller(
    #     linear_length=128,
    #     nonlinear_length=32,
    #     mu_linear=0.05,
    #     mu_nonlinear=0.005
    # )

    canceller = VolterraEchoCanceller(
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

    # Analyze results
    analyze_volterra_performance(
        canceller, far_end, near_end, echo_estimate, error_signal
    )

    return {
        'far_end': far_end,
        'near_end': near_end,
        'echo_estimate': echo_estimate,
        'error_signal': error_signal,
        'canceller': canceller
    }


if __name__ == "__main__":
    # Run the test
    results = test_volterra_echo_cancellation('C:/Users/dadab/projects/AEC/data/rec1/2')

    # Optional: Save results to audio files
    # sf.write('far_end.wav', results['far_end'], 8000)
    # sf.write('near_end_with_echo.wav', results['near_end'], 8000)
    # sf.write('echo_cancelled.wav', results['error_signal'], 8000)