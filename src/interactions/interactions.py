import numpy as np
import pandas as pd


def str_to_array(s) -> np.ndarray:
    return np.array(list(map(int, s.split("|"))))


def sort_by_interaction(row, lookup_series: pd.Series):
    impressions = str_to_array(row["impressions"])
    user_session = row['user_id'] + row['session_id']
    if user_session not in lookup_series.index:
        recommendations = str(impressions).strip('[]')
        recommendations = " ".join(recommendations.split())
        return recommendations
    interactions = lookup_series[user_session]
    recommendations = []
    # first add all interactions
    for i in interactions:
        if i in impressions:
            recommendations.append(i)
    # add all other impressions left in order
    for impr in impressions:
        if impr not in recommendations:
            recommendations.append(impr)
    # list of recommendations to single string
    recommendations = str(recommendations).strip('[]')
    recommendations = " ".join(recommendations.split())
    return recommendations


def get_lookup_series(df_source: pd.DataFrame) -> pd.Series:
    df = df_source.copy()
    df['user_session'] = df['user_id'] + df['session_id']
    df_interacted = df[(df['action_type'] == 'interaction item image') | (
            df['action_type'] == 'interaction item rating') | (
                                      df['action_type'] == 'interaction item info') | (
                                      df['action_type'] == 'interaction item deals')]
    # reverse dataframe -> the later interactions are more important
    df_interacted = df_interacted.iloc[::-1]
    df_interacted = df_interacted[['user_session', 'reference']]
    interactions_lookup = df_interacted.groupby('user_session')['reference'].apply(list)
    return interactions_lookup


def calc_recommendation(df_train: pd.DataFrame, df_target: pd.DataFrame) -> pd.DataFrame:
    """Calculate recommendations based on interactions (latest one counts more)

    The final data frame will have an impression list sorted according to the interactions

    :param df_train: Data frame with training data
    :param df_target: Data frame with target
    :return: Data frame with sorted impression list according to interactions
    """
    lookup_series = get_lookup_series(df_target)
    df_tc = df_target.copy()
    df_tc['item_recommendations'] = df_tc.apply(lambda x: sort_by_interaction(x, lookup_series), axis=1)
    df_out = df_tc[['user_id', 'session_id', 'timestamp', 'step', 'item_recommendations']]
    return df_out
