import os
import pydub
import soundfile as sf

# Convert audio files to wav, delete original audio file if `delete_orig` set
def to_wav(src, dst, delete_orig = False):
    if not os.path.isfile(src):
        raise ValueError(src + ' is not a file')
    data, samplerate = sf.read(src)
    sf.write(dst, data, samplerate, subtype='PCM_16')
    if delete_orig:
        os.remove(src)

# Convert mp3 to wav, delete original audio file if `delete_orig` set    
def mp3_to_wav(src, dst, delete_orig = False):
    if not os.path.isfile(src):
        raise ValueError(src + ' is not a file')
    sound = pydub.AudioSegment.from_file(src, format="mp3")
    sound.export(dst, format="wav")
    if delete_orig:
        os.remove(src)

# Converts all audio files in `sounds/` to wave files
def all_to_wav():
    for root, dirs, files in os.walk("sounds"):
        for file in files:
            path = os.path.join(root, file)
            (pre, ext) = os.path.splitext(path)
            if ext in [".mp3", ".aif", ".aiff", ".flac"]:
                dst = pre + ".wav"
                if ext == ".mp3":
                    mp3_to_wav(path, dst, True)
                else:
                    to_wav(path, dst, True)
 