from pathlib import Path

import click
import pandas as pd

import utils as f
import verify_submission.functions as verification
import baseline_algorithm.functions as bf
import first_impression.first_impression as fimpr
import price_ordered.price_ordered as cheapest
import price_median.price_median as prmedian
import interactions.interactions as intrctn
import current_filter.current_filter_inout as currfilterinout
import als_icf.als_icf as als_icf


current_directory = Path(__file__).absolute().parent
default_data_directory = current_directory.joinpath('..', 'data')


@click.command()
@click.option('--data-path', default=None, help='Directory for the CSV files')
@click.option('--recommender', default='interactions', help='Recommendation algorithm to be used')
def main(data_path: str, recommender: str):
    # calculate path to files
    data_directory = Path(data_path) if data_path else default_data_directory
    train_csv = data_directory.joinpath('train_split.csv')
    test_csv = data_directory.joinpath('test_split.csv')
    subm_csv = data_directory.joinpath('submission_' + recommender + '.csv')

    print(f"Reading {train_csv} ...")
    df_train = pd.read_csv(train_csv)
    print(f"Reading {test_csv} ...")
    df_test = pd.read_csv(test_csv)
    print("Identify target rows...")
    df_target = f.get_submission_target(df_test)

    if recommender == 'baseline':
        df_out = bf.calc_recommendation(df_train, df_target)
    elif recommender == 'first_impression':
        df_out = fimpr.calc_recommendation(df_train, df_target)
    elif recommender == 'cheapest':
        df_out = cheapest.calc_recommendation(df_train, df_target)
    elif recommender == 'median':
        df_out = prmedian.calc_recommendation(df_train, df_target)
    elif recommender == 'interactions':
        # use original test df as "training data"
        df_out = intrctn.calc_recommendation(df_test, df_target)
    elif recommender == 'currfilterinout':
        df_out = currfilterinout.calc_recommendation(df_train, df_target)
    elif recommender == "als_icf":
        # TODO:
        # alf_icf needs information about all users to build a model,
        # therefore you need to pass the whole training set (before splitting it)
        # to the algorithm, e.g. read train.csv into df_train instead of train_split.csv
        df_out = als_icf.calc_recommendation(df_train, df_target)
    else:
        raise Exception('algorithm ' + recommender + ' not implemented')

    if not verification.verify(df_out, df_test):
        raise Exception('submission not valid')

    print(f"Writing {subm_csv}...")
    df_out.to_csv(subm_csv, index=False)

    print("Finished calculating recommendations.")


if __name__ == '__main__':
    main()
