from data import train_data_generator, stack_gen
from model import unet
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import argparse
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MultiLabelBinarizer


print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.experimental.set_visible_devices([], 'GPU'))

if __name__ == "__main__":
    # set CUDA device to use GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # parse command line arguments
    parser = argparse.ArgumentParser(description='HMS_vision')
    parser.add_argument('-sl', '--side-len', type=int, default=256, help='side length of image')
    parser.add_argument('-bs', '--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('-en', '--epoch-num', type=int, default=30, help='epoch number')
    parser.add_argument('--smoke', type=str, default='dense', help="Which tiles to use for training; one of 'dense', 'smoke', or 'all'.")
    parser.add_argument('-ts', '--test-size', type=float, default=0.1, help="test size")
    parser.add_argument('-vs', '--val-size', type=float, default=0.1, help="validation size")
    parser.add_argument('--seed', type=int, default=np.random.randint(1, 10000), help="seed number for random")
    # create dictionary of arguments
    args = parser.parse_args()
    # set variables
    image_side_length = args.side_len
    print("image_side_length:", image_side_length)
    batch_size = args.batch_size
    print("batch_size:", batch_size)
    epoch_num = args.epoch_num
    print("epoch_num:", epoch_num)
    test_size = args.test_size
    print("test_size:", test_size)
    seed = args.seed
    print("seed:", seed)
    validation_split_rate = args.val_size
    print("validation_split_rate:", validation_split_rate)
    smoke = args.smoke
    print("smoke", smoke)

    # load csv of all tiles with specified image side length
    df = pd.read_csv('csv/data_{side_len}.csv'.format(side_len=image_side_length), index_col=['timestamp', 'num'], parse_dates=['timestamp']).sort_index()


    df['path_band1'] = df['path_band1'].str.replace('/n/holyscratch01/mickley/hms_vision_data/256x256/','/n/holyscratch01/mickley/hms_vision_data2/256x256/')
    df['path_band3'] = df['path_band3'].str.replace('/n/holyscratch01/mickley/hms_vision_data/256x256/','/n/holyscratch01/mickley/hms_vision_data2/256x256/')
    df['path_hms'] = df['path_hms'].str.replace('/n/holyscratch01/mickley/hms_vision_data/256x256/','/n/holyscratch01/mickley/hms_vision_data2/256x256/')

    # omit invalid tiles (void, on edge, night)
    data_df = df[df.not_void & df.not_edge & df.not_night & df.valid]


    # pick which tiles to use based on user specification
    # use tiles with dense smoke
    if smoke == "dense":
        data_use = data_df[data_df.is_dense]
    # use tiles with any smoke
    elif smoke == "smoke":
        data_use = data_df[data_df.hms_sum > 0]
    # use all tiles
    elif smoke == "all":
        data_use = data_df
    else:
        raise ValueError("Wrong value for --smoke")

    # pick by date procedue: split dataset by dates
    # list of date timestamps for all data
    timestamp_pick = np.unique(data_use.index.get_level_values('timestamp').date)
    # split dates into training dataset and test dataset
    date_train, date_test = train_test_split(timestamp_pick, test_size=test_size, random_state=seed)
    # create date column for dataframe
    data_df.loc[:, 'date'] = data_df.index.get_level_values('timestamp').date

    # create dataframe of tiles for training
    if smoke == "dense":
        df_train = data_df[data_df.date.isin(np.sort(date_train)) & (data_df.is_dense)]
    elif smoke == "smoke":
        df_train = data_df[data_df.date.isin(np.sort(date_train)) & (data_df.hms_sum > 0)]
    elif smoke == "all":
        df_train = data_df[data_df.date.isin(np.sort(date_train))]
    # create dataframe of tiles for test
    df_test = data_df[data_df.date.isin(np.sort(date_test))]

    # save dataframes as csv files
    df_train.to_csv("train-{smoke}-by_date-{side_len}-{seed}.csv".format(smoke=smoke, side_len=image_side_length, seed=seed))
    df_test.to_csv("test-{smoke}-by_date-{side_len}-{seed}.csv".format(smoke=smoke, side_len=image_side_length, seed=seed))

    # create image loader
    band1_gen, band3_gen, mask_gen, val_band1_gen, val_band3_gen, val_mask_gen = train_data_generator(
        dataframe=df_train,
        image_side_length=image_side_length,
        batch_size=batch_size,
        seed=seed,
        validation_split_rate=validation_split_rate
        )

    # initiate model
    model = unet(input_size=(image_side_length, image_side_length, 2))

    # save weights with least loss function value per epoch
    model_save_path = "{smoke}-by_date-{side_len}-{seed}_UNBALANCE_jaccard.hdf5".format(smoke=smoke, side_len=image_side_length, seed=seed)
    model_checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1, save_best_only=True)



    #hms weights applied in model.py


    # train!
    history = model.fit(
        x=stack_gen(band1_gen, band3_gen, mask_gen),
        epochs=epoch_num,
        verbose=2,
        steps_per_epoch=len(mask_gen),
        validation_data=stack_gen(val_band1_gen, val_band3_gen, val_mask_gen),
        validation_steps=len(val_mask_gen),
        callbacks=[model_checkpoint],
        )

    # save training history as csv file
    pd.DataFrame(history.history).to_csv("{smoke}-by_date-{side_len}-{seed}.csv".format(smoke=smoke, side_len=image_side_length, seed=seed))
