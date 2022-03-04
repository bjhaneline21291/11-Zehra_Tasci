from analyzedata import analyze_all_peaks
from clean import clean_all
from fft import compute_all_ffts
from towav import all_to_wav

clean_all()
all_to_wav()
data = compute_all_ffts()
analyze_all_peaks(data)
