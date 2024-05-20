import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def load_audio(filename):
    y, sr = librosa.load(filename, sr=None)
    return y, sr


def plot_spectrogram(y, sr, window_size=1024, hop_size=512):
    D = librosa.stft(y, n_fft=window_size, hop_length=hop_size, window='hann')

    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    plt.figure(figsize=(10, 6))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Спектрограмма (окно Ханна)')
    plt.xlabel('Время (секунды)')
    plt.ylabel('Частота (Гц)')
    plt.savefig("spectogramAAA.jpg")
    plt.close()

# Основной код
if __name__ == '__main__':
    filename = 'audio_2024-05-04_15-08-53.wav'

    y, sr = load_audio(filename)

    plot_spectrogram(y, sr, window_size=1024, hop_size=512)
