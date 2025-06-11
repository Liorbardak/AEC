import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import correlate
import librosa
import padasip as pa


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


class AdvancedInterferenceCanceller:
    def __init__(self, filter_length=512, step_size=0.3):
        self.nlms = NLMSAdaptiveFilter(filter_length, step_size)

    def find_optimal_delay(self, reference, mixed, max_delay=1000):
        """Find the delay between reference and interference in mixed signal"""
        correlation = correlate(mixed, reference, mode='full')
        delay = np.argmax(correlation) - len(reference) + 1
        return max(0, min(delay, max_delay))

    def preprocess_signals(self, reference, mixed, sample_rate):
        """Preprocess signals for better performance"""
        # Normalize signals
        reference = reference / (np.max(np.abs(reference)) + 1e-8)
        mixed = mixed / (np.max(np.abs(mixed)) + 1e-8)

        # Optional: Apply bandpass filter if you know interference frequency range
        # This step can be customized based on your specific interference characteristics

        return reference, mixed

    def cancel_interference(self, reference_signal, mixed_signal, sample_rate=44100):
        """Main interference cancellation function"""
        # Preprocess
        reference, mixed = self.preprocess_signals(reference_signal, mixed_signal, sample_rate)

        # Find optimal delay (optional - can help with performance)
        delay = self.find_optimal_delay(reference, mixed)
        print(f"Detected delay: {delay} samples ({delay / sample_rate:.3f} seconds)")

        # Apply delay compensation if significant
        if delay > 10:
            reference_delayed = np.zeros_like(reference)
            reference_delayed[delay:] = reference[:-delay]
            reference = reference_delayed

        # Apply NLMS filtering
        cleaned_signal = self.nlms.process_signals(reference, mixed)

        return cleaned_signal


def load_and_process_audio(reference_file, mixed_file):
    """Load and process real audio files"""
    try:
        # Load audio files
        reference, sr1 = librosa.load(reference_file, sr=None)
        mixed, sr2 = librosa.load(mixed_file, sr=None)

        # Ensure same sample rate
        if sr1 != sr2:
            mixed = librosa.resample(mixed, orig_sr=sr2, target_sr=sr1)
            sr2 = sr1

        # Ensure same length
        min_len = min(len(reference), len(mixed))
        reference = reference[:min_len]
        mixed = mixed[:min_len]

        print(f"Loaded signals: {len(reference)} samples at {sr1} Hz")

        # Apply interference cancellation
        canceller = AdvancedInterferenceCanceller(filter_length=512, step_size=0.1)
        cleaned = canceller.cancel_interference(reference, mixed, sr1)

        return cleaned, mixed, reference, sr1

    except Exception as e:
        print(f"Error loading audio files: {e}")
        return None, None, None, None


def demonstrate_with_synthetic_data():
    """Demonstrate with synthetic data"""
    # Parameters
    fs = 44100
    duration = 3
    t = np.linspace(0, duration, int(fs * duration))

    # Create desired signal (speech-like)
    desired = np.sin(2 * np.pi * 800 * t) * np.exp(-t / 2) + 0.3 * np.sin(2 * np.pi * 1200 * t)

    # Create interference (noise-like)
    interference = 0.8 * np.sin(2 * np.pi * 200 * t) + 0.4 * np.sin(2 * np.pi * 350 * t)

    # Reference signal (clean interference)
    reference = interference.copy()

    # Mixed signal with delay and distortion
    delay = 100  # samples
    attenuation = 0.7
    distorted_interference = np.zeros_like(interference)
    distorted_interference[delay:] = attenuation * interference[:-delay]

    # Add some nonlinear distortion
    distorted_interference = np.tanh(1.5 * distorted_interference)

    mixed = desired + distorted_interference

    # Apply cancellation
    canceller = AdvancedInterferenceCanceller(filter_length=256, step_size=0.2)
    cleaned = canceller.cancel_interference(reference, mixed, fs)

    # Plot results
    plt.figure(figsize=(15, 12))

    time_segment = slice(0, 2000)  # Show first 2000 samples

    plt.subplot(5, 1, 1)
    plt.plot(t[time_segment], desired[time_segment])
    plt.title('Original Desired Signal')
    plt.ylabel('Amplitude')

    plt.subplot(5, 1, 2)
    plt.plot(t[time_segment], reference[time_segment])
    plt.title('Reference Signal (Clean Interference)')
    plt.ylabel('Amplitude')

    plt.subplot(5, 1, 3)
    plt.plot(t[time_segment], mixed[time_segment])
    plt.title('Mixed Signal (Desired + Distorted Interference)')
    plt.ylabel('Amplitude')

    plt.subplot(5, 1, 4)
    plt.plot(t[time_segment], cleaned[time_segment])
    plt.title('Cleaned Signal (After NLMS Cancellation)')
    plt.ylabel('Amplitude')

    plt.subplot(5, 1, 5)
    plt.plot(t[time_segment], desired[time_segment], 'b-', label='Original', alpha=0.7)
    plt.plot(t[time_segment], cleaned[time_segment], 'r--', label='Recovered', alpha=0.7)
    plt.title('Comparison: Original vs Recovered')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (s)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Calculate performance metrics
    mse_before = np.mean((mixed - desired) ** 2)
    mse_after = np.mean((cleaned - desired) ** 2)
    improvement_db = 10 * np.log10(mse_before / (mse_after + 1e-10))

    print(f"MSE before cancellation: {mse_before:.6f}")
    print(f"MSE after cancellation: {mse_after:.6f}")
    print(f"Improvement: {improvement_db:.2f} dB")

    return cleaned, mixed, desired

def run_test():
    inpath = 'C:/Users/dadab/projects/AEC/data/rec1/2'

   # cleaned, mixed, reference, sr1  =     load_and_process_audio(inpath + '/mic_output.wav' ,  inpath + '/resampled_and_normalized_ai.wav')
    #
    mic_sig , mic_sr  = librosa.load(inpath + '/mic_output.wav', sr=None)
    ai_sig , ai_sr  =  librosa.load(inpath + '/resampled_and_normalized_ai.wav', sr=None)
    canceller = AdvancedInterferenceCanceller(filter_length=16, step_size=0.01)
    ai_sig = ai_sig[43000:]
    mic_sig = mic_sig[43000:]
    cleaned = canceller.cancel_interference(ai_sig, mic_sig, mic_sr)
    # # Plot results
    f = pa.filters.FilterNLMS(n=1, mu=0.1, w="random")
    cleaned, e, w = f.run(mic_sig[:,np.newaxis], ai_sig[:,np.newaxis])

    plt.figure(figsize=(15, 8))
    plt.plot(mic_sig,label='mixed')
    plt.plot(ai_sig,label='reference')
    #plt.plot(mic_sig-ai_sig,label='adapt')
    #plt.plot(mic_sig- cleaned, label='cleaned')
    plt.legend()
    plt.show()



    time_segment = slice(0, 2000)  # Show first 2000 samples

    plt.subplot(5, 1, 1)
    plt.plot(t[time_segment], desired[time_segment])
    plt.title('Original Desired Signal')
    plt.ylabel('Amplitude')

    plt.subplot(5, 1, 2)
    plt.plot(t[time_segment], reference[time_segment])
    plt.title('Reference Signal (Clean Interference)')
    plt.ylabel('Amplitude')

    plt.subplot(5, 1, 3)
    plt.plot(t[time_segment], mixed[time_segment])
    plt.title('Mixed Signal (Desired + Distorted Interference)')
    plt.ylabel('Amplitude')

    plt.subplot(5, 1, 4)
    plt.plot(t[time_segment], cleaned[time_segment])
    plt.title('Cleaned Signal (After NLMS Cancellation)')
    plt.ylabel('Amplitude')

    plt.subplot(5, 1, 5)
    plt.plot(t[time_segment], desired[time_segment], 'b-', label='Original', alpha=0.7)
    plt.plot(t[time_segment], cleaned[time_segment], 'r--', label='Recovered', alpha=0.7)
    plt.title('Comparison: Original vs Recovered')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (s)')
    plt.legend()



# Example usage
if __name__ == "__main__":
    run_test()
    # Run synthetic demonstration
    #cleaned, mixed, original = demonstrate_with_synthetic_data()

    # For real audio files, uncomment and modify paths:
    # cleaned, mixed, ref, sr = load_and_process_audio('reference.wav', 'mixed.wav')
    # if cleaned is not None:
    #     # Save cleaned audio
    #     wavfile.write('cleaned_output.wav', sr, cleaned.astype(np.float32))