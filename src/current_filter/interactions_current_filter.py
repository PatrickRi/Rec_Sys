import numpy as np
import pandas as pd
import current_filter.functions as f

###### NOT IN USE ######


def sort_by_interaction(row, lookup_series: pd.Series, metadata: pd.DataFrame):
    impressions = f.str_to_array(row["impressions"])
    filters = f.str_to_array(row['current_filters'])
    user_session = row['user_id'] + row['session_id']

    if user_session not in lookup_series.index:
        recommendations = str(impressions).strip('[]')
        recommendations = " ".join(recommendations.split())
        return recommendations
    interactions = lookup_series[user_session]
    recommendations = []
    # first add all interactions
    for i in interactions:
        if i != 'unknown' and int(i) in impressions and int(i) not in recommendations:
            recommendations.append(int(i))
    # add all other impressions left in order
    for impr in impressions:
        if impr not in recommendations:
            recommendations.append(impr)
    # list of recommendations to single string
    recommendations = str(recommendations).replace(",", "").strip('[]')
    return recommendations


def get_lookup_series(df_source: pd.DataFrame) -> pd.Series:
    df = df_source.copy()
    df['user_session'] = df['user_id'] + df['session_id']
    df_interacted = df[df['action_type'] == 'filter selection']
    # reverse dataframe -> the later interactions are more important
    df_interacted = df_interacted.iloc[::-1]
    df_interacted = df_interacted[['user_session', 'reference']]
    interactions_lookup = df_interacted.groupby('user_session')['reference'].apply(list)
    return interactions_lookup


def calc_recommendation(df_train: pd.DataFrame, df_target: pd.DataFrame) -> pd.DataFrame:
    """Calculate recommendations based on filter

    :param df_train: Data frame with training data
    :param df_target: Data frame with target
    :return: Data frame with sorted impression list based on filter
    """
    lookup_series = get_lookup_series(df_train)
    metadata = f.get_metadata_matrix()
    df_tc = df_target.copy()
    df_tc['item_recommendations'] = df_tc.apply(lambda x: sort_by_interaction(x, lookup_series, metadata), axis=1)
    df_out = df_tc[['user_id', 'session_id', 'timestamp', 'step', 'item_recommendations']]
    return df_out
