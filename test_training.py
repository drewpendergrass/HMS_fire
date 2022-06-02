import pandas as pd
import numpy as np
from data import train_data_generator
import tensorflow as tf

#conda activate mlp


print(tf.__version__)

tf.config.list_physical_devices()

print(tf.config.experimental.set_visible_devices([], 'GPU'))


from model import unet
side_len = 256
model = unet(input_size=(side_len, side_len, 2) , pretrained_weights=None, learning_rate=1e-4)


tile_dataframe = pd.read_csv('csv/data_256_mkelpcopy.csv')
tile_dataframe['path_band1'] = tile_dataframe['path_band1'].str.replace('/n/holyscratch01/mickley/hms_vision_data/256x256/','/n/holyscratch01/mickley_lab/mkelp/HMS_vision/data/')
tile_dataframe['path_band3'] = tile_dataframe['path_band3'].str.replace('/n/holyscratch01/mickley/hms_vision_data/256x256/','/n/holyscratch01/mickley_lab/mkelp/HMS_vision/data/')
tile_dataframe['path_hms'] = tile_dataframe['path_hms'].str.replace('/n/holyscratch01/mickley/hms_vision_data/256x256/','/n/holyscratch01/mickley_lab/mkelp/HMS_vision/data/')

from data import train_data_generator, stack_gen
batch_size = 32
seed = 5
validation_split_rate = 0.8
band1_gen, band3_gen, mask_gen, val_band1_gen, val_band3_gen, val_mask_gen = train_data_generator(
    dataframe=tile_dataframe,
    image_side_length=side_len,
    batch_size=batch_size,
    seed=seed,
    validation_split_rate=validation_split_rate
)




# don't actually run this cell, because it will start training
epoch_num = 10
model.fit(
    x=stack_gen(band1_gen, band3_gen, mask_gen),
    epochs=epoch_num,
    verbose=2,
    steps_per_epoch=len(mask_gen),
    validation_data=stack_gen(val_band1_gen, val_band3_gen, val_mask_gen),
    validation_steps=len(val_mask_gen))
