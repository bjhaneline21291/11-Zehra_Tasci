from collections import defaultdict
from matplotlib import pyplot as plt
import os
from scipy.fft import fft, fftfreq
from scipy.io import wavfile as wav
from scipy.signal import find_peaks


# Plot and save the Fourier transform of wav file at `path` to `dst`
# Return array of normalized amplitudes of harmonics
def compute_fft(path, dst):
    # data processing and fft calculations

    # read in wav file
    rate, data = wav.read(path)

    # only take one audio stream if multiple exist
    try:
        data = data[:, 0]
    except IndexError:
        pass

    # compute FFT, find frequencies and amplitudes
    samples = data.shape[0]
    datafft = fft(data)
    fftabs = abs(datafft)
    freqs = fftfreq(samples, 1/rate)

    frequencies = freqs[:int(freqs.size/2)]
    amplitudes = fftabs[:int(freqs.size/2)]

    # find peaks in the frequency domain (only take peaks a certain amount of max peak)
    maxpeak = max(amplitudes)
    peak_indices = find_peaks(amplitudes, prominence=0.03*maxpeak, distance=100)[0]

    peak_freqs = []
    peak_amps = []
    for i in peak_indices:
        # cut out low frequencies (noise from recording)
        if frequencies[i] >= 100:
            peak_freqs.append(frequencies[i])
            peak_amps.append(amplitudes[i])

    # plot FFT with peaks and save image
    plt.figure()
    plt.xlim([10, rate/2])
    plt.xscale("log")
    plt.grid(True)
    plt.xlabel("Frequency (Hz)")
    plt.title(path)
    plt.plot(frequencies, amplitudes)
    plt.plot(peak_freqs, peak_amps, "xr")
    plt.savefig(dst)
    plt.close()

    # normalize peaks
    norm_peak_amps = list(map(lambda peak: peak / peak_amps[0], peak_amps))
    return norm_peak_amps


# Plot and save the Fourier transform of wav file at `path` to `dst` for all files in `sounds`
# Return dictionary of normalized harmonic amplitudes per instrument
def compute_all_ffts():
    instruments = defaultdict(list)
    for root, dirs, files in os.walk("sounds"):
        for file in files:
            path = os.path.join(root, file)
            (pre, ext) = os.path.splitext(path)
            if ext in [".wav"]:
                body = os.path.sep.join(pre.split(os.path.sep)[1:])
                dst = os.path.join("images", body + ".png")
                norm_peak_amps = compute_fft(path, dst)
                instrument = root.split(os.path.sep)[1]
                instruments[instrument].append(norm_peak_amps)
    return instruments
