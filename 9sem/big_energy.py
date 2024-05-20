import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


def load_audio(filename):
    y, sr = librosa.load(filename, sr=None)
    return y, sr


def find_high_energy_moments(y, sr, dt=0.1, freq_range=(40, 50)):
    n_fft = int(sr * dt)  # Размер окна в соответствии с шагом времени
    hop_length = int(n_fft / 2)  # Шаг окна для перекрытия (половина размера окна)

    # Строим спектрограмму
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    # Вычисляем спектральную энергию в заданном частотном диапазоне
    freq_min_idx = int(freq_range[0] * n_fft / sr)
    freq_max_idx = int(freq_range[1] * n_fft / sr)
    energy_in_range = np.sum(S[freq_min_idx:freq_max_idx], axis=0)

    # Находим моменты времени с наибольшей энергией
    threshold = np.max(energy_in_range) * 0.9  # Порог (90% максимальной энергии)
    high_energy_times = np.where(energy_in_range >= threshold)[0] * hop_length / sr

    return high_energy_times


if __name__ == '__main__':
    filename = 'masha.wav'

    y, sr = load_audio(filename) # sr - частота дискретизации

    dt = 0.1  # Шаг времени (0.1 секунды)
    freq_range = (40, 50)  # Диапазон частот (40-50 Гц)

    high_energy_times = find_high_energy_moments(y, sr, dt, freq_range)

    print(f'Моменты времени с наибольшей энергией в диапазоне {freq_range[0]}-{freq_range[1]} Гц:')
    for time in high_energy_times:
        print(f' - {time:.2f} с')
