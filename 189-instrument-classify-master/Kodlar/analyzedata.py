import csv
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd

# Takes in a list of list of normalized peak amplitudes collected per trial
# and performs statistical calculations, output raw data to `dst`,
# and summary statistics to `SUMMARY.csv`
def analyze_peaks(peaks, dst, instrument):
    df = pd.DataFrame(peaks)

    # write raw data
    file_exists = os.path.isfile(dst)
    with open(dst, 'a') as f:
        headers = ['1st harmonic', '2nd harmonic', '3rd harmonic', '4th harmonic', '5th harmonic']
        writer = csv.DictWriter(f, fieldnames=headers)

        if not file_exists:
            writer.writeheader()

        df.to_csv(f, header=False, index=False)

    # write summary statistics
    with open('SUMMARY.csv', 'a') as f:
        writer = csv.writer(f)
        means = [instrument + ' means']
        means.extend(df.mean().values)
        stdevs = [instrument + ' stdevs']
        stdevs.extend(df.std().values)
        writer.writerow(means)
        writer.writerow(stdevs)

    # clean up data to plot
    means = means[1:6]
    stdevs = stdevs[1:6]
    # if stdev is not meaningful, remove value
    for i in range(len(stdevs)):
        if stdevs[i] is float('nan'):
            float('nan')
    # append `None` until length is 5
    while len(means) < 5:
        means.append(float('nan'))
    while len(stdevs) < 5:
        stdevs.append(float('nan'))
    return means, stdevs


# Takes in the output of `compute_all_ffts` and analyzes peaks
# Input is a dictionary, with keys being instruments and values
# being a 2D list of normalized peak amplitudes collected per trial
def analyze_all_peaks(data):
    # prepare for summary plot
    pltdata = {}

    # statistical summary of each instrument frequencies
    for instrument in data:
        dst = 'raw-data/' + instrument + '.csv'
        pltdata[instrument] = analyze_peaks(data[instrument], dst, instrument)

    # set width of bar
    barWidth = 1 / (len(data) + 2)

    # set position of bar on X axis and make the plot
    r = np.arange(5)
    i = 0
    for instrument in pltdata:
        plt.bar(x=r, height=pltdata[instrument][0], width=barWidth,
                yerr=pltdata[instrument][1], capsize=2,
                edgecolor='white', label=instrument)
        r = [x + barWidth for x in r]

    # add xticks on the middle of the group bars
    plt.title('Normalized Harmonic Amplitudes by Instrument')
    plt.xlabel('Harmonic', fontweight='bold')
    plt.xticks([r + 2.5 * barWidth for r in range(5)], ['1', '2', '3', '4', '5'])
    plt.legend()
    plt.savefig('SUMMARY.png')
    plt.show()
