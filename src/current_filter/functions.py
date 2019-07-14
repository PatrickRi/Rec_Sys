import pandas as pd
import numpy as np


def get_metadata_matrix() -> pd.DataFrame:
    dfm = pd.read_csv('../data/item_metadata.csv', dtype="str", sep=",", encoding="utf-8")
    dfm_stack = dfm.set_index('item_id', drop=False, append=True).properties.str.split('|', expand=True).stack()
    dfm_sparse = pd.get_dummies(dfm_stack, prefix=None, prefix_sep=None).groupby(level=1, sort=False).agg(max)
    return dfm_sparse


def str_to_array(s) -> np.ndarray:
    return np.array(list(map(int, s.split("|"))))


def str_to_str_array(s) -> np.ndarray:
    return np.array(list(s.split("|")))


def get_metadata_score(filters: np.ndarray, i: int, metadata_matrix: pd.DataFrame) -> int:
    count = 0
    for curr_filter in filters:
        try:
            if metadata_matrix.loc[str(i)][curr_filter] == 1:
                count = count + 1
        except:
            pass
    return count
