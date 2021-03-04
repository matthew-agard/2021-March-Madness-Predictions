import pandas as pd
import matplotlib.pyplot as plt


def feature_null_counts(df):
    nulls = df.isnull().sum().sort_values(ascending=False)
    return nulls[nulls > 0]


def get_null_rows(null_fills, df):
    rows = df[df[null_fills].isnull().any(axis=1)]
    return rows[['Year'] + null_fills]


def format_plot(title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.tight_layout()