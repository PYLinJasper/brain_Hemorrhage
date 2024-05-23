'''Python version 3.8.18 '''

import cv2
import pandas as pd


def init_image_stacking(csv, img_info_dir, epidural, intraparenchymal,
            subarachnoid, intraventricular, multi, subdural, subdural_1, subdural_2):
    
    # img info from each class
    file_epidural = pd.read_csv(img_info_dir + epidural + csv)
    file_intraparenchymal = pd.read_csv(img_info_dir + intraparenchymal + csv)
    file_subarachnoid = pd.read_csv(img_info_dir + subarachnoid + csv)
    file_intraventricular = pd.read_csv(img_info_dir + intraventricular + csv)
    file_multi = pd.read_csv(img_info_dir + multi + csv)
    file_subdural_1 = pd.read_csv(img_info_dir + subdural_1 + csv)
    file_subdural_2 = pd.read_csv(img_info_dir + subdural_2 + csv)
    file_subdural = pd.concat([file_subdural_1, file_subdural_2])

    file_list = [file_epidural, file_intraparenchymal, file_subarachnoid,
                file_intraventricular, file_multi, file_subdural]
    file_list_fld = [epidural, intraparenchymal, subarachnoid, 
                intraventricular, multi, subdural]
    return file_list, file_list_fld


def combine_4_image(img_dir, bbw, bw, mcw, sw, file_name):
    img_bbw = cv2.imread(img_dir + bbw + file_name, cv2.IMREAD_GRAYSCALE)
    img_bw = cv2.imread(img_dir + bw + file_name, cv2.IMREAD_GRAYSCALE)
    img_mcw = cv2.imread(img_dir + mcw + file_name, cv2.IMREAD_GRAYSCALE)
    img_sw = cv2.imread(img_dir + sw + file_name, cv2.IMREAD_GRAYSCALE)

    return [img_sw, img_bw, img_mcw, img_bbw]