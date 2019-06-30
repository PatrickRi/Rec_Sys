from pathlib import Path

import click
import pandas as pd

import verify_submission.functions as f

current_directory = Path(__file__).absolute().parent
default_data_directory = current_directory.joinpath('..', '..', 'data')


@click.command()
@click.option('--data-path', default=None, help='Directory for the CSV files')
#  @click.option('--submission-file', default='submission_popular.csv', help='Submission CSV file')
@click.option('--submission-file', default='submission_first_impression.csv', help='Submission CSV file')
@click.option('--test-file', default='test.csv', help='Test CSV file')
def main(data_path, submission_file, test_file):
    # calculate path to files
    data_directory = Path(data_path) if data_path else default_data_directory
    test_csv = data_directory.joinpath(test_file)
    subm_csv = data_directory.joinpath(submission_file)

    print(f"Reading {subm_csv} ...")
    df_subm = pd.read_csv(subm_csv)
    print(f"Reading {test_csv} ...")
    df_test = pd.read_csv(test_csv)

    if f.verify(df_subm, df_test):
        print('All checks passed')
    else:
        print('One or more checks failed')


if __name__ == '__main__':
    main()
