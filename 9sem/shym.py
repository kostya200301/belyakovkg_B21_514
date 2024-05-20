import librosa
import numpy as np


def load_audio(filename):
    y, sr = librosa.load(filename, sr=None)
    return y, sr


def estimate_snr(y):
    signal_power = np.mean(np.square(y))

    silent_part = np.concatenate((y[:sr], y[-sr:]))

    noise_power = np.mean(np.square(silent_part))

    snr = 10 * np.log10(signal_power / noise_power)

    return snr


if __name__ == '__main__':
    filename = 'masha.wav'

    y, sr = load_audio(filename)

    snr = estimate_snr(y)
    print(f'Отношение сигнал-шум (SNR): {snr:.2f} дБ')