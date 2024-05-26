'''
Python version 3.8.18
find Contour
'''

import cv2
import numpy as np
from tqdm import tqdm

from pre_config import *
from contour import init_contour_finding, find_contour


def main(args):
    '''this is the main program'''
    # data root path
    data_dir = args.root_path + '/'
    img_info_dir = data_dir + 'segmentation/'
    img_dir = data_dir + '01_Preprocessed/'
    sto_dir = '02_Contour/'
    create_folder(data_dir + sto_dir)

    file_list, file_list_fld = init_contour_finding(img_dir, epidural, intraparenchymal,
            subarachnoid, intraventricular, multi, subdural, normal)

    for file, fld in zip(file_list, file_list_fld):
        if fld == 'normal':
            break
    
        create_folder(data_dir + sto_dir + fld)
        for img_name in tqdm(file, desc=f"{fld}", unit="image"):
            # read image
            img = cv2.imread(img_dir + fld + '/' + img_name)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
            img_size0 = gray_img.shape[0]
            img_size1 = gray_img.shape[1]
            gray_img = gray_img.reshape(img_size0, img_size1, 1)

            contour = find_contour(gray_img, img_name)
    
            # Create a mask for the brightest oval-like polygon
            mask = np.zeros_like(gray_img, dtype=np.uint8)
            
            cv2.drawContours(mask, [contour], 0, 255, thickness=cv2.FILLED)

            # outside contour to be black
            result = cv2.bitwise_and(img, img, mask=mask)
            cv2.imwrite(data_dir + sto_dir + fld + '/' + img_name, result)

if __name__ == "__main__":
    ARGS = get_args()
    main(ARGS)