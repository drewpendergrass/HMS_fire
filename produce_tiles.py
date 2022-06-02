from os import path, makedirs
# from datetime import datetime, timedelta
import pandas as pd
# from dateutil.rrule import rrule, MINUTELY
from split_image import split_image
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='produce tiles')
    parser.add_argument('-sl', '--side-len', type=int, default=256, help='side length of image')
    parser.add_argument('-so', '--start-over', type=bool, default=False, help='if we should start over')
    
    args = parser.parse_args()
    SIDE_LEN = args.side_len
    START_OVER = args.start_over
    path_all_csv_path = "/n/mickley/users/ktoshima/csv/path_all.csv"
    split_csv_path = "/n/mickley/users/ktoshima/csv/data_{side_len}.csv".format(side_len=SIDE_LEN)

    
    # specify data storage path and create directory
    data_store_dir = str(SIDE_LEN) + "x" + str(SIDE_LEN)
    data_store_path = path.join("/n/holyscratch01/mickley/hms_vision_data/", data_store_dir)
    makedirs(data_store_path, exist_ok=True)

    # load csv as dataframe
    path_all_df = pd.read_csv(path_all_csv_path, index_col='timestamp', parse_dates=['timestamp'])
    
    split_df = None

    # create progress column if necessary
    progress_col = 'split_' + str(SIDE_LEN)
    if (not progress_col in path_all_df) or (START_OVER):
        path_all_df[progress_col] = False
    timestamp_list = path_all_df.sort_index().index
    path_all_df.to_csv(path_all_csv_path)

    for timestamp in timestamp_list:
        if path_all_df.at[timestamp, progress_col]:
            continue
        else:
            row = path_all_df.loc[timestamp, :]
            band1_store_path = path.join(data_store_path, 'band1', timestamp.strftime("%Y-%m-%d"))
            makedirs(band1_store_path, exist_ok=True)
            band3_store_path = path.join(data_store_path, 'band3', timestamp.strftime("%Y-%m-%d"))
            makedirs(band3_store_path, exist_ok=True)
            hms_store_path = path.join(data_store_path, 'hms', timestamp.strftime("%Y-%m-%d"))
            makedirs(hms_store_path, exist_ok=True)
            catalogue = split_image(
                side_length=SIDE_LEN,
                band1_path=row.path_band1,
                band1_store_path=band1_store_path,
                band3_path=row.path_band3,
                band3_store_path=band3_store_path,
                mask_path=row.path_hms,
                mask_store_path=hms_store_path,
                daynight_path=row.path_daynight,
                basename=timestamp.strftime("%Y-%m-%d-%H%M"),
                timestamp=timestamp
                )
            if split_df is None:
                split_df = catalogue
            else:
                split_df = pd.concat([split_df, catalogue])
            split_df.to_csv(split_csv_path)
            path_all_df.at[timestamp, progress_col] = True
            path_all_df.to_csv(path_all_csv_path)
            print(timestamp, "done")

    path_all_df.to_csv(path_all_csv_path)
