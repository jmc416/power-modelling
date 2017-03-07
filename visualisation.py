from __future__ import division

import math

from collections import Counter
from collections import defaultdict
from itertools import groupby

from matplotlib import pyplot as plt
import numpy as np

import load


def categorical_plot(feature_name, data_rows, label_rows, max_categories=30, ybottom=0.2):
    """"""
    labels_by_id = {r['id']: r[load.LABEL_NAME] for r in label_rows}

    no_churn = Counter(r[feature_name] for r in data_rows if not labels_by_id[r['id']])
    churn = Counter(r[feature_name] for r in data_rows if labels_by_id[r['id']])

    all_categories = set(no_churn.keys()).union(set(churn.keys()))

    def total_items(category):
        return churn[category] + no_churn[category]

    def churn_ratio(category):
        return churn[category] / total_items(category)

    def enough_items(category):
        min_items = 10
        return total_items(category) >= min_items

    sorted_categories = sorted(filter(enough_items, all_categories), key=churn_ratio, reverse=True)
    sorted_categories = filter(None, sorted_categories)

    num_categories = min(len(sorted_categories), max_categories)
    ind = np.arange(num_categories)    # the x locations for the groups
    width = num_categories / math.pow(num_categories, 1.2)      # the width of the bars: can also be
    # len(x) sequence

    sorted_categories = sorted_categories[:num_categories]

    churn_ratios = [churn_ratio(category) for category in sorted_categories]

    if max(len(c) for c in sorted_categories) > 10:
        ybottom = 0.5

    ax = plt.axes([0.1, ybottom, 0.8, 1 - 0.1 - ybottom])
    p1 = plt.bar(ind, churn_ratios, width, color='#d62728', axes=ax)
    p2 = plt.bar(ind, [1 - c for c in churn_ratios], width, bottom=churn_ratios, axes=ax)

    plt.ylabel('')
    plt.title('Churn ratios by %s' % feature_name)
    plt.xticks(ind, sorted_categories, rotation='vertical')
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.legend((p1[0], p2[0]), ('Churned', 'Didn''t churn'))
    plt.xlim([-1 / num_categories, num_categories + 1 / num_categories])
    plt.ylim([-.01, 1.1])

    plt.show()
