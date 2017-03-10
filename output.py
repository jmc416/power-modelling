import csv


OUTPUT_FILE = 'output.csv'


def append(id, score, churned):

    with open(OUTPUT_FILE, 'w') as f:
        f.write('%s, %s, %s' % (id, score, churned))
