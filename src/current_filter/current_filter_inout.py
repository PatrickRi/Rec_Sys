import pandas as pd

import current_filter.functions as f


def sort_by_metadata(row, metadata_matrix: pd.DataFrame):
    impressions = f.str_to_array(row["impressions"])
    if pd.isnull(row['current_filters']):
        recommendations = str(impressions).strip('[]')
        recommendations = " ".join(recommendations.split())
        return recommendations
    filters = f.str_to_str_array(row['current_filters'])
    tuples = [None] * len(impressions)
    for i in range(len(impressions)):
        tuples[i] = (impressions[i], f.get_metadata_score(filters, impressions[i], metadata_matrix))
    tuples.sort(key=lambda tup: tup[1])
    recommendations = [x[0] for x in tuples]
    recommendations = str(recommendations).strip('[]').replace(",", "")
    return recommendations


def calc_recommendation(df_train: pd.DataFrame, df_target: pd.DataFrame) -> pd.DataFrame:
    """Calculate recommendations based on current_filters

    The final data frame will have an impression list sorted according to current_filters

    :param df_train: Data frame with training data
    :param df_target: Data frame with target
    :return: Data frame with sorted impression list according to interactions
    """
    metadata = f.get_metadata_matrix()
    df_tc = df_target.copy()
    df_tc['item_recommendations'] = df_tc.apply(lambda x: sort_by_metadata(x, metadata), axis=1)
    df_out = df_tc[['user_id', 'session_id', 'timestamp', 'step', 'item_recommendations']]
    return df_out
