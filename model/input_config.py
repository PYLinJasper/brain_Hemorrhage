import os
import cv2
import ast
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from random import sample
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_folder(path):
    '''create folder'''
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
    
    return

def get_args():
    """
    Just get the command line options using argparse
    @return: Instance of argparse arguments
    """

    parser_description = 'Root path for the dcms data images'
    parser = argparse.ArgumentParser(description=parser_description)

    parser.add_argument('-r',
                        '--root',
                        # use args.column_to_parse
                        dest='root_path',
                        # accepted input type
                        type=str,
                        help='root path for images',
                        # defult value options
                        default='./dcms')

    parser.add_argument('-s',
                        '--size',
                        # use args.column_to_parse
                        dest='size',
                        # accepted input type
                        type=int,
                        help='training image size',
                        # defult value options
                        default=256)
    
    parser.add_argument('-mn',
                        '--model',
                        # use args.column_to_parse
                        dest='model',
                        # accepted input type
                        type=str,
                        help='model Name',
                        # defult value options
                        default='model')
    
    parser.add_argument('-p',
                        '--pred',
                        # use args.column_to_parse
                        dest='pred',
                        # accepted input type
                        type=int,
                        help='0: existing model, 1: train a new model',
                        # defult value options
                        default=1)

    return parser.parse_args()

def show(img, fld, mask, dia_gray_img, img_name):
    plt.figure(figsize=(10,5))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap=plt.cm.bone)
    plt.title(fld)

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap=plt.cm.bone)
    plt.title('Mask')

    plt.subplot(1, 3, 3)
    plt.imshow(dia_gray_img, cmap=plt.cm.bone)
    plt.title(img_name)
    plt.show()

def init_data_input(img_dir, epidural, intraparenchymal,
           subarachnoid, intraventricular, multi, subdural, normal):

    # img info from each class
    file_epidural = [a for a in os.listdir(img_dir + epidural) if a != '.DS_Store']
    file_intraparenchymal = [a for a in os.listdir(img_dir + intraparenchymal) if a != '.DS_Store']
    file_subarachnoid = [a for a in os.listdir(img_dir + subarachnoid) if a != '.DS_Store']
    file_intraventricular = [a for a in os.listdir(img_dir + intraventricular) if a != '.DS_Store']
    file_multi = [a for a in os.listdir(img_dir + multi) if a != '.DS_Store']
    file_subdural = [a for a in os.listdir(img_dir + subdural) if a != '.DS_Store']
    file_normal = [a for a in os.listdir(img_dir + normal) if a != '.DS_Store']

    file_list = [file_epidural, file_intraparenchymal, file_subarachnoid,
            file_intraventricular, file_multi, file_subdural, file_normal]

    file_list_fld = [epidural, intraparenchymal, subarachnoid,
                 intraventricular, multi, subdural, normal]
    return file_list, file_list_fld

def collect_coordinate(area):
    coordinates = []
    for coordinate in area:
        if isinstance(coordinate, str) == False:
            coordinate = [i for i in coordinate.values()]
            coordinates.append(coordinate)
    return coordinates

def collect_area_coor(areas_coordinates, area):
    coordinates = []
    coordinates = collect_coordinate(area)

    if len(coordinates):
        areas_coordinates.append(coordinates)
    return areas_coordinates

def resolve(label):
    nested_list = ast.literal_eval(label)
    areas = np.array(nested_list, dtype=object)[0]
    areas = areas.replace('[]', '')

    return areas

def collect_areas(label, flag):

    if flag:
        label = resolve(label)
        # no annotation found
        if len(label) == 0:
            return False

    nested_list = ast.literal_eval(label)
    areas = np.array(nested_list, dtype=object)

    if len(areas) == 0:
        return False

    # print(areas)
    areas_coordinates = []
    if areas.ndim > 1:
        for area in areas:
            areas_coordinates = collect_area_coor(areas_coordinates, area)
    else:
        if isinstance(areas[0], list):
            # fake 1-D array
            for area in areas:
                areas_coordinates = collect_area_coor(areas_coordinates, area)
        else:
            # real 1-D array
            areas_coordinates = collect_area_coor(areas_coordinates, areas)

    return areas_coordinates

def mask_diagnose(areas_coordinates, diagnose):
    # Create a black image with the same shape as the input image
    mask = np.zeros_like(diagnose, dtype=np.uint8)
    background = np.zeros_like(diagnose, dtype=np.uint8)

    for coordinates in areas_coordinates:
        # Some with empty list
        if len(coordinates):
            vertices_scaled = (coordinates * np.array(diagnose.shape[:2])).astype(np.int32)
            vertices_scaled = vertices_scaled.reshape((-1, 1, 2))

            # Draw the filled polygon on the black background
            cv2.drawContours(mask, [vertices_scaled], 0, (255, 255, 255), thickness=cv2.FILLED)
            cv2.polylines(background, [vertices_scaled], isClosed=True,
                                        color=(255, 255, 255), thickness=3)

    # Invert the black and white colors in the background
    background = cv2.bitwise_not(background)

    # Combine the original image and the inverted background using bitwise_and
    diagnose = cv2.bitwise_and(background, diagnose)

    return diagnose, mask

def label_proven(ml, al):
    areas_coordinates = collect_areas(ml, 0)
    # no cl and ml -> use all labels
    if areas_coordinates == False:
        areas_coordinates = collect_areas(al, 1)

        if areas_coordinates == False:
            areas_coordinates = []

    return areas_coordinates

def nor_res(img, diagnose, img_size, mask):
    # some pic is not square or same size
    img = tf.image.resize(img, [img_size, img_size]).numpy()
    dia_img = tf.image.resize(diagnose, [img_size, img_size]).numpy()
    mask = tf.image.resize(mask, [img_size, img_size]).numpy()
    # Normalize
    img = img / 255.0
    # turn 3-cahnnel img into 1-channel gray scale img to reduce size
    dia_gray_img = cv2.cvtColor(dia_img, cv2.COLOR_BGR2GRAY)
    dia_gray_img = dia_gray_img.reshape(img_size, img_size, 1)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = mask.reshape(img_size, img_size, 1)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = gray_img.reshape(img_size, img_size, 1)

    # return img, gray_img, dia_img, mask
    return img, gray_img, dia_gray_img, mask

def one_hot(img_name, img_class, label_pd):

    # store data label
    onehot = label_pd.loc[img_name[:-4]][img_class].to_numpy()
    # toggle any for nomal
    index = img_class.index('any')
    onehot[index] = (onehot[index] + 1 ) % 2

    return onehot

def count_per(multi_class, img_label, y_img_sftm):
    '''Calculate porpotion'''
    if multi_class:
        model_type = 'sftm'
    else:
        model_type = 'sgmd'

    y_img_df = pd.DataFrame(y_img_sftm)

    y_img_df.columns = img_label
    class_count = [[] for _ in range(len(img_label))]
    class_num = 0

    for col in y_img_df:
        # class: 1
        class_count[class_num] = y_img_df[col].value_counts()[1]
        class_num += 1

    class_count = np.array(class_count)

    name = []
    type_count = []
    percent_count = []

    for a, b in zip(img_label, class_count):
        name.append(a)
        type_count.append(b)
        percent_count.append(str(round(b/class_count.sum()*100))+'%')

    df = pd.DataFrame()
    df['Name'] = name
    df[model_type] = type_count
    df['%'] = percent_count
    df = df.set_index('Name')

    return df

def data_generate(rpath, img_size, multi_class):
    '''called by main to generate training data from raw'''
    # constant
    epidural = 'epidural'
    intraparenchymal = 'intraparenchymal'
    subarachnoid = 'subarachnoid'
    intraventricular = 'intraventricular'
    multi = 'multi'
    subdural = 'subdural'
    subdural_1 = 'subdural_1'
    subdural_2 = 'subdural_2'
    normal = 'normal'
    label_file = 'labels'
    csv = '.csv'
    jpg = '.jpg'


    img_root_dir = rpath + '/' + '02_Contour/'
    img_info_dir = rpath + '/' + 'segmentation/'

    file_list, file_list_fld = init_data_input(img_root_dir, epidural, intraparenchymal,
            subarachnoid, intraventricular, multi, subdural, normal)

    img_label = file_list_fld.copy()
    img_label.remove('multi')
    img_class = img_label.copy()
    img_class = list(map(lambda x: x.replace('normal', 'any'), img_class))

    label_pd = pd.read_csv(img_info_dir + label_file + csv, index_col='Image')

    # multi_class = False

    if multi_class:
        # img_data_sftm = []
        grey_img_data_sftm = []
        label_img_data_sftm = []
        # mask_sftm = []
        y_img_sftm = []

    else:
        # img_data_sgmd = []
        grey_img_data_sgmd = []
        label_img_data_sgmd = []
        # mask_sgmd = []
        y_img_sgmd = []


    for img_names, fld in zip(file_list, file_list_fld):

        if fld == 'intraventricular':
            file = pd.read_csv(img_info_dir + fld + csv)
            file = file[['Origin', 'ROI', 'All Annotations']]

        elif fld != 'normal':
            # not intraventricular and normal
            if fld == 'subdural':
                file1 = pd.read_csv(img_info_dir + fld + '_1' + csv)
                file2 = pd.read_csv(img_info_dir + fld + '_2' + csv)
                file = pd.concat([file1, file2])
            else:
                file = pd.read_csv(img_info_dir + fld + csv)

            file = file[['Origin', 'Majority Label', 'Correct Label', 'All Labels']]

        img_dir = img_root_dir + fld + '/'

        for img_name in tqdm(img_names, desc=f"{fld}", unit="image"):
            # read image
            img_path = img_dir + img_name
            img = cv2.imread(img_path)
            # grey_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if fld == 'normal':
                mask = np.zeros_like(img, dtype=np.uint8)
                # normalize
                img, gray_img, dia_img, mask = nor_res(img, img, img_size, mask)


                y_one_hot = one_hot(img_name, img_class, label_pd)

                if multi_class:
                    # img_data_sftm.append(img)
                    grey_img_data_sftm.append(gray_img)
                    label_img_data_sftm.append(dia_img)
                    # mask_sftm.append(mask)
                    y_img_sftm.append(y_one_hot)

                else:
                    # img_data_sgmd.append(img)
                    grey_img_data_sgmd.append(gray_img)
                    label_img_data_sgmd.append(dia_img)
                    # mask_sgmd.append(mask)
                    y_img_sgmd.append(y_one_hot)

            else:

                labels = file[file.Origin == img_name]

                if fld == 'intraventricular':
                    iter = zip(labels['ROI'], labels['All Annotations'], labels['Origin'])

                else:
                    iter = zip(labels['Correct Label'], labels['Majority Label'],
                                    labels['All Labels'])

                # print(img_name)
                # find the label
                all_areas_coordinates = []
                for cl, ml, al in iter:

                    # have correct label
                    if isinstance(cl, str):
                        areas_coordinates = collect_areas(cl, 0)

                        if len(areas_coordinates) == 0:
                            areas_coordinates = label_proven(ml, al)

                    # have majority label
                    elif isinstance(ml, str):
                        # find coordinates
                        areas_coordinates = label_proven(ml, al)

                    all_areas_coordinates += areas_coordinates

                if len(all_areas_coordinates) > 0:
                    diagnose, mask = mask_diagnose(all_areas_coordinates, img)
                    # Normalize and resize
                    img, gray_img, dia_img, mask = nor_res(img, diagnose, img_size, mask)

                    # store training data
                    y_one_hot = one_hot(img_name, img_class, label_pd)

                    if multi_class:
                        if fld != 'multi':
                            # img_data_sftm.append(img)
                            grey_img_data_sftm.append(gray_img)
                            label_img_data_sftm.append(dia_img)
                            # mask_sftm.append(mask)
                            y_img_sftm.append(y_one_hot)

                    else:
                        # img_data_sgmd.append(img)
                        grey_img_data_sgmd.append(gray_img)
                        label_img_data_sgmd.append(dia_img)
                        # mask_sgmd.append(mask)
                        y_img_sgmd.append(y_one_hot)

                # break
        # break

    if multi_class:
        # img_data_sftm = np.array(img_data_sftm)
        # grey_img_data_sftm = np.array(img_data_sftm)
        grey_img_data_sftm = np.array(grey_img_data_sftm)
        label_img_data_sftm = np.array(label_img_data_sftm)
        # mask_sftm = np.array(mask_sftm)
        y_img_sftm = np.array(y_img_sftm)

    else:
        # img_data_sgmd = np.array(img_data_sgmd)
        # grey_img_data_sgmd = np.array(img_data_sgmd)
        grey_img_data_sgmd = np.array(grey_img_data_sgmd)
        label_img_data_sgmd = np.array(label_img_data_sgmd)
        # mask_sgmd = np.array(mask_sgmd)
        y_img_sgmd = np.array(y_img_sgmd)

    # summary of pretraining data
    print('\nType Accumulation for input data')
    if multi_class:
        print(count_per(multi_class, img_label, y_img_sftm))
    else:
        print(count_per(multi_class, img_label, y_img_sgmd))


    '''Image Stacking '''
    # stack together
    if multi_class:
        gray_label_data = np.stack([grey_img_data_sftm, label_img_data_sftm], axis=1)
        y_img = y_img_sftm

    else:
        gray_label_data = np.stack([grey_img_data_sgmd, label_img_data_sgmd], axis=1)
        y_img = y_img_sgmd

    print('\nData Shape')
    gray_label_data.shape

    '''Train Test Split'''
    # split the data: test
    x_train_raw, x_test, y_train_raw, y_test = train_test_split(gray_label_data, y_img,
                                                        test_size=0.2, random_state=10)

    # split the data: train, val
    x_train, x_val, y_train, y_val = train_test_split(x_train_raw, y_train_raw,
                                                        test_size=0.2, random_state=10)

    X_train = x_train[:, 0, :, :]
    X_train_label = x_train[:, 1, :, :]
    X_test = x_test[:, 0, :, :]
    X_test_label = x_test[:, 1, :, :]
    X_val = x_val[:, 0, :, :]
    X_val_label = x_val[:, 1, :, :]

    # dataset data count
    train_cnt = X_train.shape[0]
    test_cnt = X_test.shape[0]
    val_cnt = X_val.shape[0]

    print(f'\nTrain data: {X_train.shape}')
    print(f'Test data: {X_test.shape}')
    print(f'Val data: {X_val.shape}')

    print('\n__________________________________________')
    print('\nType Accumulation for raw training data')
    df = count_per(multi_class, img_label, y_train)
    print(df)

    '''Image Augmentation'''
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Create empty lists to store augmented data and labels
    augmented_data = []
    augmented_labels = []

    num_data = df.max().values[0]

    # Iterate over each label
    for label_idx in range(len(img_label)):  # Assuming num_classes is the number of classes

        # Get indices of samples with the current label
        label_indices = np.where(y_train[:, label_idx] == 1)[0]

        # Calculate the number of samples needed for augmentation
        samples_needed = num_data - len(label_indices)

        # Randomly select samples from the original data with the current label
        selected_samples = np.random.choice(label_indices, samples_needed, replace=True)

        # Augment the selected samples
        for sample_idx in selected_samples:
            x = X_train[sample_idx]
            y = y_train[sample_idx]

            # Reshape the image to (1, height, width, channels) as flow requires a 4D array
            x = x.reshape((1,) + x.shape)

            # Generate augmented samples
            for batch, _ in datagen.flow(x, np.zeros(1), batch_size=1):  # Use np.zeros(1) as a placeholder for y
                augmented_data.append(batch[0])
                augmented_labels.append(y)

                # Break the loop to avoid infinite augmentation
                if len(augmented_data) >= samples_needed:
                    break

    # Convert the lists to NumPy arrays
    augmented_data = np.array(augmented_data)
    augmented_labels = np.array(augmented_labels)

    # Concatenate the augmented data with the original data
    X_train_agmt = np.concatenate((X_train, augmented_data), axis=0)
    y_train_agmt = np.concatenate((y_train, augmented_labels), axis=0)

    print('\n__________________________________________')
    print('\nType Accumulation for Final training data')
    print(count_per(multi_class, img_label, y_train_agmt))

    return X_train_agmt, y_train_agmt, X_test, y_test, X_val, y_val, img_label
