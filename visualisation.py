from __future__ import division

import math

from collections import Counter
from collections import defaultdict
from itertools import groupby

from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity

import load
import preprocessing


def categorical_plot(feature_name, data_rows, label_rows, max_categories=30,
                     show=False, save=True):
    """Plot the ratio of labels for each category"""
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
    width = num_categories / math.pow(num_categories, 1.2) if num_categories else 1     # the
    # width of the bars: can
    # also be
    # len(x) sequence

    sorted_categories = sorted_categories[:num_categories]

    churn_ratios = [churn_ratio(category) for category in sorted_categories]

    ax = plt.axes()
    p1 = plt.bar(ind, churn_ratios, width, color='#d62728', axes=ax)
    p2 = plt.bar(ind, [1 - c for c in churn_ratios], width, bottom=churn_ratios, axes=ax)

    plt.ylabel('')
    plt.title('Churn ratios by %s' % feature_name)
    plt.xticks(ind, sorted_categories, rotation='vertical')
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.legend((p1[0], p2[0]), ('Churned', 'Didn''t churn'))
    plt.xlim([-1 / num_categories, num_categories + 1 / num_categories])
    plt.ylim([-.01, 1.1])

    plt.tight_layout()

    show_or_save(feature_name, save, show)


def show_or_save(feature_name, save, show):
    if save:
        plt.savefig(feature_name + '.png')
    if show:
        plt.show()


def label_map(label_rows):
    labels_by_id = {r['id']: r[load.LABEL_NAME] for r in label_rows}
    return labels_by_id


def build_churned(label_rows):
    labels_by_id = label_map(label_rows)

    def churned(row):
        return labels_by_id[row['id']]
    return churned


def continuous_plot(feature_name, data_rows, label_rows, log_x=True, bandwidth=0.2, show=False,
                    save=True, is_date=False, strip_zeros=False):
    """Plot the distribution of a feature, coloured by label"""
    date_factor = 100000000

    churned = build_churned(label_rows)

    churned_samples = []
    no_churned_samples = []
    for i, r in enumerate(data_rows):
        if not r[feature_name]:
            continue
        value = np.float64(r[feature_name])
        if is_date:
            value = value / date_factor
        if log_x or strip_zeros:
            if value <= 0:
                continue
        if log_x:
                value = math.log(value, 10)
        if churned(r):
            churned_samples.append(value)
        else:
            no_churned_samples.append(value)

    churned_samples = np.array(churned_samples)[:, np.newaxis]
    no_churned_samples = np.array(no_churned_samples)[:, np.newaxis]

    def kde(samples):
        return KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(samples)

    churned_dist = kde(churned_samples)
    no_churned_dist = kde(no_churned_samples)

    X_plot = np.linspace(min(min(churned_samples), min(no_churned_samples)),
                         max(max(churned_samples), max(no_churned_samples)),
                         1000)[1:, np.newaxis]

    fig, ax = plt.subplots()

    log_dens = churned_dist.score_samples(X_plot)
    ax.plot(X_plot[:, 0], np.exp(log_dens), '-', label='Churned', color='r')

    log_dens = no_churned_dist.score_samples(X_plot)
    ax.plot(X_plot[:, 0], np.exp(log_dens), '-', label='No Churn')

    ax.legend(loc='upper left')
    plt.title('The distribution of %s' % feature_name)
    plt.xlabel('log %s' % feature_name if log_x else feature_name)
    plt.ylabel('Density')

    if is_date:
        xticks = ax.get_xticks()
        plt.xticks(xticks,
                   [preprocessing.format_timestamp(value * date_factor) for value in xticks],
                   rotation='vertical')

    ax.plot(no_churned_samples[:, 0], -0.01 - 0.05 * np.random.random(no_churned_samples.shape[0]),
            '+b', alpha=0.1)
    ax.plot(churned_samples[:, 0], -0.01 - 0.05 * np.random.random(churned_samples.shape[0]),
            '.r', alpha=0.3)

    plt.tight_layout()
    show_or_save(feature_name, save, show)


def timeseries_plot(feature_name, timeseries_rows, label_rows, save=True, show=False):
    """Plot all the timeseries for a feature, coloured by label"""
    churned = build_churned(label_rows)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for row in timeseries_rows:
        try:
            x, y = preprocessing.make_xy(feature_name, row)
            ax.plot(x, y, '%ss-' % 'r' if churned(row) else 'b',
                    alpha=0.2 if not churned(row) else 0.4, marker=None)
        except:
            print row[feature_name]

    plt.xticks(x, [preprocessing.format_timestamp(t) for t in x], rotation='vertical')
    plt.tight_layout()
    show_or_save(feature_name, save, show)


