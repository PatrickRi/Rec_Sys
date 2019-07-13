import statistics as stats

import numpy as np
import pandas as pd


def str_to_array(s) -> np.ndarray:
    return np.array(list(map(int, s.split("|"))))


def sorted_indices_by_distance_to_median(prices: np.ndarray) -> list:
    tuples = [None] * len(prices)
    median = stats.median(prices)
    for i in range(len(prices)):
        tuples[i] = (i, abs(median - prices[i]))
    tuples.sort(key=lambda tup: tup[1])
    return [x[0] for x in tuples]


def sort_by_price_alternating(row):
    impr = str_to_array(row["impressions"])
    prices = str_to_array(row["prices"])
    sorted_indices = sorted_indices_by_distance_to_median(prices)
    recommendations = [None] * len(impr)
    for i in range(len(impr)):
        recommendations[i] = impr[sorted_indices[i]]
    # list of recommendations to single string
    recommendations = str(recommendations).strip('[]').replace(",", "")
    return recommendations


def calc_recommendation(df_train: pd.DataFrame, df_target: pd.DataFrame) -> pd.DataFrame:
    """Calculate recommendations based on prices (median first, then alternating cheaper and more expensive)

    The final data frame will have an impression list sorted according to the price.

    :param df_train: Data frame with training data
    :param df_target: Data frame with target
    :return: Data frame with sorted impression list according to price
    """
    df_tc = df_target.copy()
    df_tc['item_recommendations'] = df_tc.apply(sort_by_price_alternating, axis=1)
    df_out = df_tc[['user_id', 'session_id', 'timestamp', 'step', 'item_recommendations']]
    return df_out
