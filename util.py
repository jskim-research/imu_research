import sklearn as sk
import sklearn.metrics.pairwise
import numpy as np
from sklearn.decomposition import PCA


def pca(data: np.ndarray) -> np.ndarray:
    """
    Args:
        data: T-length time series data with N features which has shape (T, N)
    Returns:
        data after PCA which has shape (T, 1)
    """
    pca = PCA(n_components=1)
    data = pca.fit_transform(data)
    return data


# modified from https://stackoverflow.com/questions/33650371/recurrence-plot-in-python
def recurrence_plot(data: np.ndarray, clip: int) -> np.ndarray:
    """
    Args:
        data: T-length time series data with N features which has shape (T, N)
        clip: maximum value for distance
    """
    # pairwise_distances 함수 설명:
    #   default = use euclidean distance
    #   d[i, j] == np.sqrt(np.sum((s[i] - s[j]) ** 2))
    self_dist = sk.metrics.pairwise.pairwise_distances(data)
    self_dist[self_dist > clip] = clip  # 일정 값 이상 clipping
    # Z = squareform(d)
    return self_dist


def stft(data: np.ndarray, fft_size: int) -> np.ndarray:
    """
    Args:
        data: T-length time series data which has shape (T,)
        fft_size:
    Returns:
        spectrogram which has shape (ceil(len(data) / np.float32(hop_size), fft_size/2)
    """
    overlap_fac = 0.5

    hop_size = np.int32(np.floor(fft_size * (1 - overlap_fac)))
    total_segments = np.int32(np.ceil(len(data) / np.float32(hop_size)))

    window = np.hanning(fft_size)
    result = np.zeros((fft_size // 2, total_segments), dtype=np.float32)

    for i in range(total_segments):
        current_hop = hop_size * i
        segment = data[current_hop:current_hop + fft_size]

        if len(segment) < fft_size:
            segment = np.concatenate([segment, np.array([0] * (fft_size - len(segment)))])
        windowed = segment * window
        spectrum = np.fft.fft(windowed) / fft_size

        autopower = np.abs(spectrum * np.conj(spectrum))
        autopower = np.log10(autopower)

        result[:, i] = autopower[:fft_size // 2]
    return result
