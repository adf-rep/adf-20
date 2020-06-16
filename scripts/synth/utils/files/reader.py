import numpy as np


def read_series(input_path):
    series = {}
    with open(input_path) as f:
        content = f.readlines()

        for line in content:
            v = line.split(',')
            label = v[0]

            vals = v[1:]
            vals[-1] = vals[-1].replace('\n', '')
            vals = np.array(vals)

            series[label] = vals.astype(np.float)

    return series
