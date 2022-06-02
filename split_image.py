import numpy as np
from PIL import Image, ImageFile
from os import path
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True

# On given file path, return an array of tiled images with given side length
def tile_image(file_path, side_length):
    # load image file as grayscale into numpy array
    img = Image.open(file_path).convert('L')
    img_array = np.array(img)
    # split horizontally then stack on a new axis
    h_split = np.split(img_array, np.arange(side_length, img_array.shape[0], side_length), axis=0)[:-1]
    h_stack = np.stack(h_split, axis=0)
    # split vertically then stack on a new axis
    v_split = np.split(h_stack, np.arange(side_length, img_array.shape[1], side_length), axis=2)[:-1]
    v_stack = np.stack(v_split, axis=1)
    # resulting array is (#rows of images, #columns of images, side_length, side_length)
    return v_stack

def on_edge(tile, range_num):
    top = slice(0, 0+range_num)
    btm = slice(-1-range_num, -1)
    return (np.all(tile[top,top]==0) and np.all(tile[btm,btm]!=0)) or (np.all(tile[top,btm]==0) and np.all(tile[btm,top]!=0))

def is_not_edge(band1_tile, band3_tile):
    tile1_is_not_edge = not on_edge(band1_tile, 30)
    tile3_is_not_edge = not on_edge(band3_tile, 30)
    return tile1_is_not_edge and tile3_is_not_edge

def is_not_night(daynight_tile):
    return (daynight_tile==0).sum() > daynight_tile.size*0.9

def is_not_void(band1_tile, band3_tile):
    return band1_tile.sum() > 0 or band3_tile.sum() > 0

def is_valid_tile(band1_tile, band3_tile, daynight_tile):
    not_void = band1_tile.sum() > 0 or band3_tile.sum() > 0
    not_edge = is_not_edge(band1_tile, band3_tile)
    not_night = (daynight_tile==0).sum() > daynight_tile.size*0.9
    return not_void and not_edge and not_night

def save_tile(tile_array, save_directory, filename):
    im = Image.fromarray(tile_array)
    impath = path.join(save_directory, filename)
    im.save(impath)
    return impath

def save_tiles(band1_array, band1_directory,
               band3_array, band3_directory,
               mask_array, mask_directory,
               daynight_array,
               basename, timestamp):
    df_dict = {
        "timestamp": [],
        "num": [],
        "path_band1": [],
        "path_band3": [],
        "path_hms": [],
        "hms_sum": [],
        "is_dense": [],
        "not_void": [],
        "not_edge": [],
        "not_night": [],
        "row_"+tile_row: [],
        "col_"+tile_column: []
    }
    # iterate over all tiles by row and column
    tile_row, tile_column = band1_array.shape[:2]
    for i in range(tile_row):
        for j in range(tile_column):
            band1_tile = band1_array[i, j, :, :]
            band3_tile = band3_array[i, j, :, :]
            hms_tile = mask_array[i, j, :, :]
            daynight_tile = daynight_array[i, j, :, :]
            k = np.ravel_multi_index((i, j), dims=(tile_row, tile_column))
            # check if the tile is not on border and not on night zone
            not_void = is_not_void(band1_tile, band3_tile)
            not_edge = is_not_edge(band1_tile, band3_tile)
            not_night = is_not_night(daynight_tile)
            filename = basename + "_" + str(k).zfill(4) + '.png'
            # construct image from array and save
            # band1 tile
            band1path = save_tile(band1_tile, band1_directory, filename)
            # band3 tile
            band3path = save_tile(band3_tile, band3_directory, filename)
            # construct mask from array and save
            mskpath = save_tile(hms_tile, mask_directory, filename)
            hms_sum = hms_tile.sum()
            is_dense = (hms_tile == 27).sum() > 0
            df_dict["timestamp"].append(timestamp)
            df_dict["num"].append(k)
            df_dict["path_band1"].append(band1path)
            df_dict["path_band3"].append(band3path)
            df_dict["path_hms"].append(mskpath)
            df_dict["hms_sum"].append(hms_sum)
            df_dict["is_dense"].append(is_dense)
            df_dict["not_void"].append(not_void)
            df_dict["not_edge"].append(not_edge)
            df_dict["not_night"].append(not_night)
            df_dict["row_"+tile_row].append(i)
            df_dict["col_"+tile_column].append(j)
    return df_dict
                
def split_image(
    side_length,
    band1_path,
    band1_store_path,
    band3_path,
    band3_store_path,
    mask_path,
    mask_store_path,
    daynight_path,
    basename,
    timestamp):
    band1_arr = tile_image(band1_path, side_length)
    band3_arr = tile_image(band3_path, side_length)
    mask_arr = tile_image(mask_path, side_length)
    daynight_arr = tile_image(daynight_path, side_length)
    df_dict = save_tiles(band1_arr, band1_store_path,
                      band3_arr, band3_store_path,
                      mask_arr, mask_store_path,
                      daynight_arr, 
                      basename, timestamp)
    return pd.DataFrame.from_dict(df_dict).set_index(["timestamp", "num"], drop=True, append=False, inplace=False)
