'''
Python version 3.8.18
Softmax input
'''

import pandas as pd

from input_config import *

def main(args):
    '''main function'''

    rpath = args.root_path
    img_size = args.size
    data_generate(rpath, img_size)
    # img_root_dir = args.root_path + '/' + '02_Contour/'
    # img_info_dir = args.root_path + '/' + 'segmentation/'

    # file_list, file_list_fld = init_data_input(img_root_dir, epidural, intraparenchymal,
    #         subarachnoid, intraventricular, multi, subdural, normal)

    # img_label = file_list_fld.copy()
    # img_label.remove('multi')
    # img_class = img_label.copy()
    # img_class = list(map(lambda x: x.replace('normal', 'any'), img_class))

    # label_pd = pd.read_csv(img_info_dir + label_file + csv, index_col='Image')

    # img_size = args.size

    # print(input_func(img_size))


if __name__ == "__main__":
    ARGS = get_args()
    main(ARGS)