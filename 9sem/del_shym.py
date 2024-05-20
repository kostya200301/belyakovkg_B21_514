import librosa
import soundfile as sf
import numpy as np
from scipy.signal import savgol_filter


def load_audio(filename):
    y, sr = librosa.load(filename, sr=None)
    return y, sr


def filter_audio_with_savgol(y, window_length=101, polyorder=3):
    # фильтр Савицкого-Голея
    y_filtered = savgol_filter(y, window_length=window_length, polyorder=polyorder)
    return y_filtered


def save_audio(y, sr, filename):
    sf.write(filename, y, sr)


if __name__ == '__main__':
    filename = 'masha.wav'

    y, sr = load_audio(filename)

    window_length = 101  # Размер окна
    polyorder = 3  # Степень полинома

    # Применение фильтра Савицкого-Голея
    y_filtered = filter_audio_with_savgol(y, window_length=window_length, polyorder=polyorder)

    output_filename = filename.replace('.wav', '_filtered.wav')

    save_audio(y_filtered, sr, output_filename)

    print(f'Фильтрованный аудио файл сохранен как: {output_filename}')
