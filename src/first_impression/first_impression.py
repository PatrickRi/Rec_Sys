import math
import pandas as pd
import numpy as np

GR_COLS = ["user_id", "session_id", "timestamp", "step"]


def remove_pipe_from_string(s) -> str:
    """Convert pipe separated string to string separated by a single whitespace."""

    if isinstance(s, str):
        out = s.replace("|", " ")
    elif math.isnan(s):
        out = ''
    else:
        raise ValueError("Value must be either string of nan")
    return out


def calc_recommendation(df_train: pd.DataFrame, df_target: pd.DataFrame) -> pd.DataFrame:
    """Calculate recommendations based on the first impression in the list

    The final data frame will have an impression list sorted according to the number of clicks per item in a reference data frame.

    :param df_train: Data frame with training data
    :param df_target: Data frame with target
    :return: Data frame with sorted impression list according to impression order
    """
    df_out = df_target[['user_id', 'session_id', 'timestamp','step', 'impressions']].copy()
    df_out['item_recommendations'] = df_out['impressions'].apply(remove_pipe_from_string)
    # df_target['item_recommendations'] = df_target['impressions'].apply(remove_pipe_from_string).copy()[
    #     'user_id', 'session_id', 'timestamp',
    #     'step', 'item_recommendations']
    df_out.drop('impressions', axis=1, inplace=True)
    return df_out
