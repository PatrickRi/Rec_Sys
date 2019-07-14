import implicit  # The Cython library
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.preprocessing import MinMaxScaler


# This script is based on: https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe

def calc_recommendation(df_train: pd.DataFrame, df_target: pd.DataFrame) -> pd.DataFrame:
    # current_directory = Path(__file__).absolute().parent
    # default_data_directory = current_directory.joinpath('..', 'data')

    # train_csv = default_data_directory.joinpath('train_split.csv')

    # df_train = pd.read_csv(train_csv)

    data = df_train.copy()
    data = data[data['reference'].astype(str).str.isdigit()]

    data = data.groupby(['user_id', 'reference']).size().reset_index().rename(columns={0: 'count'})

    # Create a numeric user_id and artist_id column
    data['user_id_cat'] = data['user_id'].astype("category")
    data['ref_cat'] = data['reference'].astype("category")
    data['user_id_cat'] = data['user_id_cat'].cat.codes
    data['ref_cat'] = data['ref_cat'].cat.codes

    user_lookup = data[['user_id_cat', 'user_id']].drop_duplicates()
    ref_lookup = data[['ref_cat', 'reference']].drop_duplicates()

    # user_id = "00RL8Z82B2Z1"

    # user_id_cat = user_lookup.user_id_cat.loc[user_lookup.user_id == user_id].iloc[0]

    # The implicit library expects data as a item-user matrix so we
    # create two matricies, one for fitting the model (item-user)
    # and one for recommendations (user-item)
    sparse_item_user = sparse.csr_matrix((data['count'].astype(float), (data['ref_cat'], data['user_id_cat'])))
    sparse_user_item = sparse.csr_matrix((data['count'].astype(float), (data['user_id_cat'], data['ref_cat'])))

    # Initialize the als model and fit it using the sparse item-user matrix
    model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20)

    # Calculate the confidence by multiplying it by our alpha value.
    alpha_val = 15
    data_conf = (sparse_item_user * alpha_val).astype('double')

    # Fit the model
    model.fit(data_conf)

    # ---------------------
    # FIND SIMILAR ITEMS
    # ---------------------

    # Find the 10 most similar to Jay-Z
    item_id = 147068  # Jay-Z
    n_similar = 10

    # Get the user and item vectors from our trained model
    user_vecs = model.user_factors
    item_vecs = model.item_factors

    # Calculate the vector norms
    item_norms = np.sqrt((item_vecs * item_vecs).sum(axis=1))

    # Calculate the similarity score, grab the top N items and
    # create a list of item-score tuples of most similar artists
    scores = item_vecs.dot(item_vecs[item_id]) / item_norms
    top_idx = np.argpartition(scores, -n_similar)[-n_similar:]
    similar = sorted(zip(top_idx, scores[top_idx] / item_norms[item_id]), key=lambda x: -x[1])

    # Print the names of our most similar artists
    # for item in similar:
    # idx, score = item
    # print(data.reference.loc[data.ref_cat == idx].iloc[0])

    # ------------------------------
    # CREATE USER RECOMMENDATIONS
    # ------------------------------

    def recommend(user_id, sparse_user_item, user_vecs, item_vecs, num_items=10):
        """The same recommendation function we used before"""

        user_interactions = sparse_user_item[user_id, :].toarray()

        user_interactions = user_interactions.reshape(-1) + 1
        user_interactions[user_interactions > 1] = 0

        rec_vector = user_vecs[user_id, :].dot(item_vecs.T).toarray()

        min_max = MinMaxScaler()
        rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1, 1))[:, 0]
        recommend_vector = user_interactions * rec_vector_scaled

        item_idx = np.argsort(recommend_vector)[::-1][:num_items]

        references = []
        scores = []

        for idx in item_idx:
            # references.append(data.reference.loc[data.ref_cat == idx].iloc[0])
            references.append(str(ref_lookup.reference.loc[ref_lookup.ref_cat == idx].iloc[0]))
            scores.append(recommend_vector[idx])

        # recommendations = pd.DataFrame({'reference': references, 'score': scores})
        recommendations = references

        return recommendations

    # Get the trained user and item vectors. We convert them to
    # csr matrices to work with our previous recommend function.
    user_vecs = sparse.csr_matrix(model.user_factors)
    item_vecs = sparse.csr_matrix(model.item_factors)

    """"# Create recommendations for user with id 00RL8Z82B2Z1
    user_id = "00RL8Z82B2Z1"

    user_id_cat = user_lookup.user_id_cat.loc[user_lookup.user_id == user_id].iloc[0]

    recommendations = recommend(user_id_cat, sparse_user_item, user_vecs, item_vecs)

    print(recommendations)"""

    def calc_rec(row):
        user_id_cat = user_lookup.user_id_cat.loc[user_lookup.user_id == row["user_id"]].iloc[0]

        recommendations = recommend(user_id_cat, sparse_user_item, user_vecs, item_vecs)

        return " ".join(recommendations)

    df_target_copy = df_target.copy()

    df_target_copy["item_recommendations"] = df_target_copy.apply(calc_rec, axis=1)
    return df_target_copy[['user_id', 'session_id', 'timestamp', 'step', 'item_recommendations']]
