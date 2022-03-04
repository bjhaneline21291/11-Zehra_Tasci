instrument-classify
===================

A project to classify instruments by sound using Fourier analysis

### Usage

If running for the first time, run `pip3 install -r requirements.txt`.

Place all audio files in the appropriate folder in `sounds/`. Then, run `python3 main.py`. All AIFF, FLAC, and MP3 files will automatically be converted to WAV files. The Fourier transform plots will be output in the appropriate folders in `images/`, and the detected peaks will be marked in red. The program will detect and normalize the FFT peaks by the amplitude of the first harmonic, and take the average of the normalized values for each instrument. Normalized data are output in `raw-data`, and summary CSV file and plot are output into this directory as `SUMMARY.csv` and `SUMMARY.png`.

### Source files

| File              | Description                 |
| ----------------- | --------------------------- |
| main.py           | Main program                |
| fft.py            | Compute FFT of audio file   |
| analyzedata.py    | Does data analysis of FFTs  |
| to-wav.py         | Convert audio files to WAV  |
| clean.py          | Clean file system           |
