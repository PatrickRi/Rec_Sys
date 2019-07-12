import numpy as np
import pandas as pd


def str_to_array(s) -> np.ndarray:
    return np.array(list(map(int, s.split("|"))))


def sort_by_price(row):
    impr = str_to_array(row["impressions"])
    prices = str_to_array(row["prices"])
    tuples = [None] * len(impr)
    for i in range(len(impr)):
        tuples[i] = (impr[i], prices[i])
    tuples.sort(key=lambda tup: tup[1])
    recommendations = [x[0] for x in tuples]
    # list of recommendations to single string
    recommendations = str(recommendations).strip('[]').replace(",", "")
    return recommendations


def calc_recommendation(df_train: pd.DataFrame, df_target: pd.DataFrame) -> pd.DataFrame:
    """Calculate recommendations based on prices (cheapest first)

    The final data frame will have an impression list sorted according to the price.

    :param df_train: Data frame with training data
    :param df_target: Data frame with target
    :return: Data frame with sorted impression list according to price
    """
    df_tc = df_target.copy()
    df_tc['item_recommendations'] = df_tc.apply(sort_by_price, axis=1)
    df_out = df_tc[['user_id', 'session_id', 'timestamp', 'step', 'item_recommendations']]
    return df_out
