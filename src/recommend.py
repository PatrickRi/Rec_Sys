from pathlib import Path

import click
import pandas as pd

import utils as f
import verify_submission.functions as verification
import baseline_algorithm.functions as bf
import first_impression.first_impression as fimpr

current_directory = Path(__file__).absolute().parent
default_data_directory = current_directory.joinpath('..', 'data')


@click.command()
@click.option('--data-path', default=None, help='Directory for the CSV files')
@click.option('--recommender', default='first_impression', help='Recommendation algorithm to be used')
def main(data_path: str, recommender: str):
    # calculate path to files
    data_directory = Path(data_path) if data_path else default_data_directory
    train_csv = data_directory.joinpath('train.csv')
    test_csv = data_directory.joinpath('test.csv')
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
    else:
        raise Exception('algorithm ' + recommender + ' not implemented')

    if not verification.verify(df_out, df_test):
        raise Exception('submission not valid')

    print(f"Writing {subm_csv}...")
    df_out.to_csv(subm_csv, index=False)

    print("Finished calculating recommendations.")


if __name__ == '__main__':
    main()
