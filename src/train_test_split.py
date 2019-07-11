from pathlib import Path

import click
import pandas as pd

from sklearn.model_selection import train_test_split

current_directory = Path(__file__).absolute().parent
default_data_directory = current_directory.joinpath('..', 'data')


@click.command()
@click.option('--data-path', default=None, help='Directory for the CSV files')
@click.option('--original-train-csv', default='train.csv', help='Train data file to create the split from')
def main(data_path, original_train_csv):
    # calculate path to files
    data_directory = Path(data_path) if data_path else default_data_directory
    original_train_csv = data_directory.joinpath(original_train_csv)
    train_split_csv = data_directory.joinpath('train_split.csv')
    test_split_csv = data_directory.joinpath('test_split.csv')

    print("Reading original train data ...")
    df_train = pd.read_csv(original_train_csv)

    # Do not shuffle the sets to keep sessions together
    train, test = train_test_split(df_train, test_size=0.2, shuffle=False)

    # Create ground truth file (Based on the script provided by group 3)
    print("Creating ground truth file ...")
    ground_truth_csv = data_directory.joinpath('ground_truth.csv')

    ground_truth_df = test.loc[(~test.duplicated(["user_id", "session_id"], keep="last")) & (
            test.action_type == "clickout item")]

    print("Writing ground truth to disk ...")
    ground_truth_df.to_csv(ground_truth_csv, index=False)

    # Remove last clickout reference of user/session combinations
    print("Removing clickout references from test set ...")
    test.loc[(~test.duplicated(["user_id", "session_id"], keep="last")) & (
            test.action_type == "clickout item"), "reference"] = ""

    print("Writing train/test split to disk ...")
    train.to_csv(train_split_csv, index=False)
    test.to_csv(test_split_csv, index=False)


if __name__ == '__main__':
    main()
