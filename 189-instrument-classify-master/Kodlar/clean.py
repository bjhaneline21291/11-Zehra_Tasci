import csv
import os

# delete all files in a directory, NOT including subdirectories
def clean(dir):
    for root, dirs, files in os.walk(dir):
        for file in files:
            path = os.path.join(root, file)
            if os.path.isfile(path):
                os.remove(path)


# delete all files in `data` and `images`, not including subdirectories
def clean_all():
    clean('raw-data')
    clean('images')
    with open('SUMMARY.csv', 'w') as f:
        headers = ['instrument', '1st harmonic', '2nd harmonic', '3rd harmonic', '4th harmonic', '5th harmonic']
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
