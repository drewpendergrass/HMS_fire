import numpy as np
import tensorflow as tf
from os import listdir, path

# map HMS shading to smoke density label of 0-4, only used internally
def map_HMS_mask(array, classify_level='density', threshold=3):
    if classify_level == 'density':
        replace_dict = {0: 0, 5: 1, 16: 2, 27: 3}
    elif classify_level == 'binary':
        if threshold == 1:
            replace_dict = {0: 0, 5: 1, 16: 1, 27: 1}
        elif threshold == 2:
            replace_dict = {0: 0, 5: 0, 16: 1, 27: 1}
        elif threshold == 3:
            replace_dict = {0: 0, 5: 0, 16: 0, 27: 1}
        else:
            raise ValueError("Wrong threshold for HMS")
    def replace_label(value):
        return replace_dict[value]
    return np.vectorize(replace_label)(array)

def train_data_generator(
    train_path=None,
    band1_dir=None, band3_dir=None, mask_dir=None,
    dataframe=None,
    image_side_length=512,
    batch_size=8,
    seed=None,
    shuffle=True,
    flip=True,
    validation_split_rate=0.1,
    classify_level='density',
    hms_threshold=3,
    validate_filenames=False
    ):
    """
    Return instances of ImageDataGenerator for training/validation dataset.
    Args:
    - train_path: path to the dirctory containing subdirectories of images
    - image_dir: name of the subdir containing image (GOES)
    - mask_dir: name of the subdir containing mask (HMS)
    - dataframe: dataframe containing image paths
    - image_side_length: the length of the side of images
    - batch_size: size of batch for each step of training/validation
    - seed: need it since we would like to shuffle in the same manner for image and mask data generator
    - validation_split_rate: how much data we save for validation
    - classify_level: either 'density' or 'binary' to set HMS classification level
    Return:
    ImageDataGenerator for training image, training mask, validation image, validation mask
    """
    # configuration for each instance of image and mask ImageDataGenerator
    # necessary augumentation should be configured here
    image_gen_args = dict(
        rescale=1./255,
        fill_mode='reflect',
        horizontal_flip=flip,
        vertical_flip=flip,
        validation_split=validation_split_rate
        )
    mask_gen_args = dict(
        fill_mode='reflect',
        horizontal_flip=flip,
        vertical_flip=flip,
        preprocessing_function=lambda arr: map_HMS_mask(array=arr, classify_level=classify_level, threshold=hms_threshold),
        validation_split=validation_split_rate
        )

    # create instances of ImageDataGenerator based on the configuration
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**image_gen_args)
    mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**mask_gen_args)

    if train_path is not None:
        band1_path = path.join(train_path, band1_dir)
        band3_path = path.join(train_path, band3_dir)
        hms_path = path.join(train_path, mask_dir)
        subdir_classes = listdir(band1_path)
        # create generator that loads images from train_path/dir_name
        # create generator for training dataset
        band1_generator = image_datagen.flow_from_directory(
            directory=band1_path,
            target_size=(image_side_length, image_side_length),
            color_mode="grayscale",
            classes=subdir_classes,
            class_mode=None,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            subset='training'
            )
        band3_generator = image_datagen.flow_from_directory(
            directory=band3_path,
            target_size=(image_side_length, image_side_length),
            color_mode="grayscale",
            classes=subdir_classes,
            class_mode=None,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            subset='training'
            )
        mask_generator = mask_datagen.flow_from_directory(
            directory=hms_path,
            target_size=(image_side_length, image_side_length),
            color_mode="grayscale",
            classes=subdir_classes,
            class_mode=None,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            subset='training'
            )
        # create generator for validation dataset
        val_band1_generator = image_datagen.flow_from_directory(
            directory=band1_path,
            target_size=(image_side_length, image_side_length),
            color_mode="grayscale",
            classes=subdir_classes,
            class_mode=None,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            subset='validation'
            )
        val_band3_generator = image_datagen.flow_from_directory(
            directory=band3_path,
            target_size=(image_side_length, image_side_length),
            color_mode="grayscale",
            classes=subdir_classes,
            class_mode=None,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            subset='validation'
            )
        val_mask_generator = mask_datagen.flow_from_directory(
            directory=hms_path,
            target_size=(image_side_length, image_side_length),
            color_mode="grayscale",
            classes=subdir_classes,
            class_mode=None,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            subset='validation'
            )

    elif dataframe is not None:
        band1_generator = image_datagen.flow_from_dataframe(
            dataframe,
            directory=None,
            x_col="path_band1",
            y_col=None,
            target_size=(image_side_length, image_side_length),
            color_mode="grayscale",
            classes=None,
            class_mode=None,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            subset="training",
            validate_filenames=validate_filenames,
            )
        band3_generator = image_datagen.flow_from_dataframe(
            dataframe,
            directory=None,
            x_col="path_band3",
            y_col=None,
            target_size=(image_side_length, image_side_length),
            color_mode="grayscale",
            classes=None,
            class_mode=None,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            subset="training",
            validate_filenames=validate_filenames,
            )
        mask_generator = mask_datagen.flow_from_dataframe(
            dataframe,
            directory=None,
            x_col="path_hms",
            y_col=None,
            target_size=(image_side_length, image_side_length),
            color_mode="grayscale",
            classes=None,
            class_mode=None,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            subset="training",
            validate_filenames=validate_filenames,
            )
        # create generator for validation dataset
        val_band1_generator = image_datagen.flow_from_dataframe(
            dataframe,
            directory=None,
            x_col="path_band1",
            y_col=None,
            target_size=(image_side_length, image_side_length),
            color_mode="grayscale",
            classes=None,
            class_mode=None,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            subset="validation",
            validate_filenames=validate_filenames,
            )
        val_band3_generator = image_datagen.flow_from_dataframe(
            dataframe,
            directory=None,
            x_col="path_band3",
            y_col=None,
            target_size=(image_side_length, image_side_length),
            color_mode="grayscale",
            classes=None,
            class_mode=None,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            subset="validation",
            validate_filenames=validate_filenames,
            )
        val_mask_generator = mask_datagen.flow_from_dataframe(
            dataframe,
            directory=None,
            x_col="path_hms",
            y_col=None,
            target_size=(image_side_length, image_side_length),
            color_mode="grayscale",
            classes=None,
            class_mode=None,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            subset="validation",
            validate_filenames=validate_filenames,
            )

    else:
        raise ValueError("Either train_path or dataframe should be given.")

    return band1_generator, band3_generator, mask_generator, \
           val_band1_generator, val_band3_generator, val_mask_generator

# coupling image/mask generator, since keras.model.fit() does not take zipped generators
def stack_gen(band1_gen, band3_gen, mask_gen, auto_encoder=False):
    for band1, band3, mask in zip(band1_gen, band3_gen, mask_gen):
        if auto_encoder:
            yield (np.concatenate((band1, band3), axis=-1), np.concatenate((band1, band3), axis=-1))
        else:
            yield (np.concatenate((band1, band3), axis=-1), mask)
