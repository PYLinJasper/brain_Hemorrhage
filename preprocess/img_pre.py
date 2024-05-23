'''Python version 3.8.18 '''

import os
import cv2
import numpy as np
from tqdm import tqdm
from random import sample

from config import *
from stack_combine import combine_4_image, init_image_stacking


sto_dir = '01_Preprocessed/'
create_folder(data_dir + sto_dir)

file_list, file_list_fld = init_image_stacking(csv, img_info_dir, epidural, intraparenchymal,
           subarachnoid, intraventricular, multi, subdural, subdural_1, subdural_2)

print("Combine images' four windows to one picture and save them in preprocessed")

bbw = '/brain_bone_window/'
bw = '/brain_window/'
mcw = '/max_contrast_window/'
sw = '/subdural_window/'

for file, fld in zip(file_list, file_list_fld):
    '''Combine images' four windows to one picture and save them in folder preprocessed'''
    create_folder(data_dir + sto_dir + fld)

    for file_name in tqdm(file.Origin.unique(), desc=f"{fld}", unit="image"):
        # read all four CT scan
        image_list = combine_4_image(img_dir + fld, bbw, bw, mcw, sw, file_name)
        # turn into 4-channel image
        stack_img = np.stack(image_list, axis=-1)

        cv2.imwrite(data_dir + sto_dir + fld + '/' + file_name, stack_img)


# for Normal image
image_names = sample([a for a in os.listdir(img_dir + 'normal/brain_window') if a != '.DS_Store'], 1000)

for file_name in tqdm(image_names, desc=f"normal", unit="image"):
    '''normal brain CT sampling'''
    create_folder(data_dir + sto_dir + 'normal')

    # read all four CT scan
    image_list = combine_4_image(img_dir + 'normal/', bbw, bw, mcw, sw, file_name)
    # turn into 4-channel image
    stack_img = np.stack(image_list, axis=-1)
    
    cv2.imwrite(data_dir + sto_dir + 'normal/' + file_name, stack_img)