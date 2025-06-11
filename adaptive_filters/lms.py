import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import soundfile as sf


class LMSEchoCanceller:
    """
    LMS Adaptive Filter for Echo Cancellation

    This implementation uses the Least Mean Squares algorithm to adaptively
    estimate and cancel echo in audio signals.
    """

    def __init__(self, filter_length=512, step_size=0.01):
        """
        Initialize the LMS echo canceller

        Args:
            filter_length (int): Length of the adaptive filter (number of taps)
            step_size (float): Learning rate (mu) for LMS algorithm
        """
        self.N = filter_length  # Filter length
        self.mu = step_size  # Step size (learning rate)
        self.w = np.zeros(self.N)  # Adaptive filter coefficients
        self.x_buffer = np.zeros(self.N)  # Input signal buffer

    def update(self, x_n, d_n):
        """
        Update the adaptive filter for one sample

        Args:
            x_n (float): Reference signal (far-end signal)
            d_n (float): Desired signal (near-end + echo)

        Returns:
            tuple: (output_signal, error_signal, estimated_echo)
        """
        # Shift the buffer and add new sample
        self.x_buffer[1:] = self.x_buffer[:-1]
        self.x_buffer[0] = x_n

        # Calculate estimated echo
        y_n = np.dot(self.w, self.x_buffer)

        # Calculate error (echo-cancelled signal)
        e_n = d_n - y_n

        # Update filter coefficients using LMS algorithm
        self.w += self.mu * e_n * self.x_buffer

        return e_n, y_n

    def process_signals(self, reference_signal, microphone_signal):
        """
        Process entire signals for echo cancellation

        Args:
            reference_signal (array): Far-end reference signal
            microphone_signal (array): Near-end microphone signal (with echo)

        Returns:
            tuple: (echo_cancelled_signal, estimated_echo_signal, error_history)
        """
        N_samples = min(len(reference_signal), len(microphone_signal))
        echo_cancelled = np.zeros(N_samples)
        estimated_echo = np.zeros(N_samples)
        error_history = np.zeros(N_samples)

        for n in range(N_samples):
            e_n, y_n = self.update(reference_signal[n], microphone_signal[n])
            echo_cancelled[n] = e_n
            estimated_echo[n] = y_n
            error_history[n] = e_n ** 2  # Squared error for monitoring convergence

        return echo_cancelled, estimated_echo, error_history

    def reset(self):
        """Reset the adaptive filter"""
        self.w = np.zeros(self.N)
        self.x_buffer = np.zeros(self.N)


def create_echo_simulation(clean_signal, echo_delay=0.1, echo_gain=0.3, fs=16000):
    """
    Create a simulated echo scenario for testing

    Args:
        clean_signal (array): Original clean signal
        echo_delay (float): Echo delay in seconds
        echo_gain (float): Echo attenuation factor
        fs (int): Sampling frequency

    Returns:
        tuple: (reference_signal, microphone_signal_with_echo)
    """
    delay_samples = int(echo_delay * fs)

    # Create echo signal
    echo_signal = np.zeros_like(clean_signal)
    if delay_samples < len(clean_signal):
        echo_signal[delay_samples:] = echo_gain * clean_signal[:-delay_samples]

    # Reference signal (far-end)
    reference_signal = clean_signal

    # Microphone signal (near-end speech + echo)
    # For simulation, we'll use a simple near-end signal
    near_end_signal = 0.5 * clean_signal  # Simulated near-end speech
    microphone_signal = near_end_signal + echo_signal

    return reference_signal, microphone_signal


def demonstrate_echo_cancellation():
    """Demonstrate the LMS echo canceller with a test signal"""

    # Parameters
    fs = 8000  # Sampling frequency
    duration = 5.0  # Signal duration in seconds
    f1, f2 = 300, 1200  # Frequencies for test signal

    # Generate test signals
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    clean_signal = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
    clean_signal += 0.1 * np.random.randn(len(clean_signal))  # Add some noise

    # Create echo simulation
    reference_signal, microphone_signal = create_echo_simulation(
        clean_signal, echo_delay=0.05, echo_gain=0.4, fs=fs
    )


    # add delay
    inpath = 'C:/Users/dadab/projects/AEC/data/rec1/2'
    microphone_signal, sr = sf.read(inpath + '/mic_output.wav')
    ref, sr = sf.read(inpath + '/resampled_and_normalized_ai.wav')


    gp = 1

    reference_signal = microphone_signal[gp:]
    microphone_signal = microphone_signal[:-gp]

    # Initialize LMS echo canceller
    echo_canceller = LMSEchoCanceller(filter_length=256, step_size=0.001)

    # Process signals
    echo_cancelled, estimated_echo, error_history = echo_canceller.process_signals(
        reference_signal, microphone_signal
    )

    # Plot results
    plt.figure(figsize=(15, 12))

    # Time vector for plotting
    t_plot = np.arange(len(microphone_signal)) / fs

    # Original signals
    plt.subplot(4, 1, 1)
    plt.plot(t_plot, reference_signal, label='Reference (Far-end)', alpha=0.7)
    plt.plot(t_plot, microphone_signal, label='Microphone (with echo)', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Input Signals')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Echo cancellation result
    plt.subplot(4, 1, 2)
    plt.plot(t_plot, microphone_signal, label='Original (with echo)', alpha=0.7)
    plt.plot(t_plot, echo_cancelled, label='Echo cancelled', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Echo Cancellation Result')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Estimated echo
    plt.subplot(4, 1, 3)
    plt.plot(t_plot, estimated_echo, label='Estimated echo', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Estimated Echo Signal')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Learning curve (error convergence)
    plt.subplot(4, 1, 4)
    error_db = 10 * np.log10(np.maximum(error_history, 1e-10))
    plt.plot(t_plot, error_db)
    plt.xlabel('Time (s)')
    plt.ylabel('Squared Error (dB)')
    plt.title('LMS Convergence (Learning Curve)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.figure()
    plt.plot(echo_canceller.w)
    plt.show()

    # Calculate performance metrics
    echo_reduction = calculate_echo_reduction(microphone_signal, echo_cancelled)
    print(f"Echo Reduction: {echo_reduction:.2f} dB")

    return echo_canceller, echo_cancelled, estimated_echo


def calculate_echo_reduction(original_signal, cancelled_signal):
    """Calculate echo reduction in dB"""
    original_power = np.mean(original_signal ** 2)
    cancelled_power = np.mean(cancelled_signal ** 2)

    if cancelled_power > 0:
        reduction_db = 10 * np.log10(original_power / cancelled_power)
    else:
        reduction_db = float('inf')

    return reduction_db


# Advanced LMS variants
class NLMSEchoCanceller(LMSEchoCanceller):
    """Normalized LMS (NLMS) for better convergence"""

    def __init__(self, filter_length=512, step_size=0.5, regularization=1e-6):
        super().__init__(filter_length, step_size)
        self.eps = regularization  # Small regularization constant

    def update(self, x_n, d_n):
        # Shift buffer
        self.x_buffer[1:] = self.x_buffer[:-1]
        self.x_buffer[0] = x_n

        # Calculate estimated echo
        y_n = np.dot(self.w, self.x_buffer)

        # Calculate error
        e_n = d_n - y_n

        # Normalized step size
        x_power = np.dot(self.x_buffer, self.x_buffer) + self.eps
        normalized_mu = self.mu / x_power

        # Update coefficients
        self.w += normalized_mu * e_n * self.x_buffer

        return e_n, y_n


# Example usage and testing
if __name__ == "__main__":
    print("LMS Adaptive Echo Cancellation Demo")
    print("=" * 40)

    # Run demonstration
    canceller, cancelled_signal, estimated_echo = demonstrate_echo_cancellation()

    print("\nDemo completed! Check the plots to see the results.")
    print("\nKey Features:")
    print("- Adaptive filter learns echo path characteristics")
    print("- Real-time processing capability")
    print("- Convergence monitoring through error signal")
    print("- Support for both standard LMS and NLMS variants")
