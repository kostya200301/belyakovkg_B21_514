import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter
from scipy.io import wavfile
import librosa
import matplotlib.pyplot as plt
import os


def extract_main_tone(input_path):
    samples, sample_rate = librosa.load(input_path)
    chroma_stft = librosa.feature.chroma_stft(y=samples, sr=sample_rate)
    pitches, _ = librosa.piptrack(y=samples, sr=sample_rate, S=chroma_stft)
    main_tone = np.argmax(pitches)

    return main_tone

def extract_frequencies(input_path):
    samples, sample_rate = librosa.load(input_path, sr=None)
    decibels = librosa.amplitude_to_db(np.abs(librosa.stft(samples)), ref=np.max)
    frequencies = librosa.fft_frequencies(sr=sample_rate)
    mean_spectrum = np.mean(decibels, axis=1)

    min_index = np.argmax(mean_spectrum > -80)
    max_index = len(mean_spectrum) - np.argmax(mean_spectrum[::-1] > -80) - 1

    min_frequency = frequencies[min_index]
    max_frequency = frequencies[max_index]

    return max_frequency, min_frequency

def detect_formants(frequencies, times, spectrogram):
    time_window = int(0.1 * len(times))
    frequency_window = int(50 / (frequencies[1] - frequencies[0]))
    filtered_spectrogram = maximum_filter(spectrogram, size=(frequency_window, time_window))

    peak_mask = (spectrogram == filtered_spectrogram)
    peak_values = spectrogram[peak_mask]
    peak_frequencies = frequencies[peak_mask.any(axis=1)]

    top_indices = np.argsort(peak_values)[-3:]
    top_formant_frequencies = peak_frequencies[top_indices]

    return list(top_formant_frequencies)

def create_spectrogram(samples, sample_rate, output_path):
    frequencies, times, spectrogram = signal.spectrogram(
        samples, sample_rate, scaling='spectrum', window='hann')
    log_spectrogram = np.log10(spectrogram)

    plt.pcolormesh(times, frequencies, log_spectrogram, shading='auto')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.savefig(output_path)

    return frequencies, times, spectrogram
def process_audio_file(input_path: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output')
    output_path = os.path.join(output_dir, os.path.splitext("spec_" + os.path.basename(input_path))[0] + ".png")
    title = os.path.splitext(os.path.basename(input_path))[0]

    sample_rate, samples = wavfile.read(input_path)
    frequencies, times, spectrogram = create_spectrogram(samples, sample_rate, output_path)
    max_frequency, min_frequency = extract_frequencies(input_path)
    formants = detect_formants(frequencies, times, spectrogram)
    main_tone = extract_main_tone(input_path)

    print(f"{title}\n")
    print(f"Max frequency: {max_frequency}\n")
    print(f"Min frequency: {min_frequency}\n")
    print(f"Main tone: {main_tone}\n")
    print(f"Strongest formants: {formants}\n")


process_audio_file("input/aaaa.wav")
process_audio_file("input/iiii.wav")
process_audio_file("input/gafgaf.wav")
