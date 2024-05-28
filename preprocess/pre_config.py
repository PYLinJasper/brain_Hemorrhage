'''Constant'''

import os
import argparse

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
                        default='./dcms/')

    return parser.parse_args()