from __future__ import division

import math

from collections import Counter
from collections import defaultdict
from itertools import groupby

from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity

import load


def categorical_plot(feature_name, data_rows, label_rows, max_categories=30, ybottom=0.2):
    """"""
    churned = build_churned(label_rows)

    no_churn = Counter(r[feature_name] for r in data_rows if not churned(r))
    churn = Counter(r[feature_name] for r in data_rows if churned(r))

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


def label_map(label_rows):
    labels_by_id = {r['id']: r[load.LABEL_NAME] for r in label_rows}
    return labels_by_id


def build_churned(label_rows):
    labels_by_id = label_map(label_rows)
    def churned(row):
        return labels_by_id[row['id']]
    return churned


def continuous_plot(feature_name, data_rows, label_rows, log_x=True):
    """"""
    churned = build_churned(label_rows)
    num_samples = len(data_rows)

    churned_samples = np.zeros([num_samples, 1])
    no_churned_samples = np.zeros([num_samples, 1])
    num_churned = 0
    for i, r in enumerate(data_rows):
        value = np.float64(r[feature_name])
        if log_x:
            if value <= 0:
                continue
            else:
                value = math.log(value, 10)
        if churned(r):
            churned_samples[i] = value
            num_churned += 1
        else:
            no_churned_samples[i] = value

    churned_samples = churned_samples[:num_churned]
    no_churned_samples = no_churned_samples[:(num_samples - num_churned)]

    churned_dist = kde(churned_samples)
    no_churned_dist = kde(no_churned_samples)

    X_plot = np.linspace(max(min(min(churned_samples), min(no_churned_samples)), 0),
                         max(max(churned_samples), max(no_churned_samples)),
                         1000)[1:, np.newaxis]



    fig, ax = plt.subplots()

    log_dens = churned_dist.score_samples(X_plot)
    ax.plot(X_plot[:, 0], np.exp(log_dens), '-', label='Churned')

    log_dens = no_churned_dist.score_samples(X_plot)
    ax.plot(X_plot[:, 0], np.exp(log_dens), '-', label='No Churn')

    ax.legend(loc='upper left')
    ax.plot(churned_samples[:, 0], -0.005 - 0.01 * np.random.random(churned_samples.shape[0]), '+r')
    ax.plot(no_churned_samples[:, 0], -0.005 - 0.01 * np.random.random(no_churned_samples.shape[0]),
            '.b')
    plt.show()

def kde(churned_samples):
    return KernelDensity(kernel='gaussian', bandwidth=0.2).fit(churned_samples)
