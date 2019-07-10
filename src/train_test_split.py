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

    df_train = pd.read_csv(original_train_csv)

    # Do not shuffle the sets to keep sessions together
    train, test = train_test_split(df_train, test_size=0.2, shuffle=False)

    train.to_csv(train_split_csv, index=False)

    test.to_csv(test_split_csv, index=False)

    # Create ground truth file
    create_ground_truth(data_path, test_split_csv)

    # Remove last clickout reference of user/session combinations
    # TODO: reeeaaally slow ...
    mask = test["action_type"] == "clickout item"
    masked_test = test[mask]

    last_row = None
    last_index = None

    for index, row in masked_test.iterrows():
        if last_row is not None:
            if last_row["user_id"] == row["user_id"] and last_row["session_id"] == row["session_id"]:
                pass
            else:
                test.loc[last_index, "reference"] = ""

        last_row = row
        last_index = index

    # test_split_ref_removed_csv = data_directory.joinpath('test_split_reference_removed.csv')
    test.to_csv(test_split_csv, index=False)


# Based on the script provided by group 3
def create_ground_truth(data_path, test_file):

    # calculate path to files
    data_directory = Path(data_path) if data_path else default_data_directory
    test_csv = data_directory.joinpath(test_file)
    gt_csv = data_directory.joinpath('ground_truth.csv')

    #print('Reading files...')
    df_test = pd.read_csv(test_csv)
    mask_click_out = df_test["action_type"] == "clickout item"
    df_clicks = df_test[mask_click_out]

    mask_ground_truth = df_clicks["reference"].notnull()
    df_gt = df_clicks[mask_ground_truth]

    df_gt.to_csv(gt_csv, index=False)

    #print('finished')


if __name__ == '__main__':
    main()
