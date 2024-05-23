'''Constant'''

import os

epidural = 'epidural'
intraparenchymal = 'intraparenchymal'
subarachnoid = 'subarachnoid'
intraventricular = 'intraventricular'
multi = 'multi'
subdural = 'subdural'
subdural_1 = 'subdural_1'
subdural_2 = 'subdural_2'
normal = 'normal'

csv = '.csv'
jpg = '.jpg'

# data root path
data_dir = './dcms/'
img_info_dir = data_dir + 'segmentation/'
img_dir = data_dir + 'renders/'

def create_folder(path):
    '''create folder'''
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
    
    return