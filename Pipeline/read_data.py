# Others
import os #Systems interactions
from pathlib import Path
from os import path
from skimage.io import imread
import sys

def Read_inputs_pipeline(path):
    print('Reading images...')

    data = dict()
    data['description'] = 'Dataset of the pipeline inputs.'
    data['number'] = [] # Counter
    data['filename'] = [] # File name. Useless but stored
    data['image'] = [] # Here is the image stored

    inputs_path = path

    counter = 0
    for file in os.listdir(inputs_path):
        if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg')or file.endswith('.pgm')or file.endswith('.pbm')or file.endswith('.bmp'):
            im = imread(os.path.join(inputs_path, file))
            data['filename'].append(file)
            data['number'].append(counter)
            data['image'].append(im)
            print('\t\tImage', data['number'][counter], ':', data['filename'][counter])
            counter += 1
    print("\tNumber of images read: ", counter)
    print("* Done reading images.")
    return data


def Create_directory(counter):
    current_path = os.path.dirname(os.path.abspath(__file__))

    directory_name = 'Results'
    fpath = os.path.join(current_path, directory_name)
    # Creating directory
    if path.exists(fpath) == False:
        os.mkdir(fpath)

    directory_name = 'Results/Image_{}'.format(counter)
    fpath = os.path.join(current_path, directory_name)
    # Creating directory
    if path.exists(fpath) == False:
        os.mkdir(fpath)

    directory_name = directory_name + '/Image_{}_lines'.format(counter)
    fpath = os.path.join(current_path, directory_name)
    # Creating directory
    if path.exists(fpath) == False:
        os.mkdir(fpath)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def Model_data():
    labels = ['Alef', 'Ayin', 'Bet', 'Dalet', 'Gimel', 'He', 'Het', 'Kaf', 'Kaf-final', 'Lamed', 'Mem', 'Mem-medial',
              'Nun-final', 'Nun-medial', 'Pe', 'Pe-final', 'Qof', 'Resh', 'Samekh',
              'Shin', 'Taw', 'Tet', 'Tsadi-final', 'Tsadi-medial', 'Waw', 'Yod', 'Zayin']

    char_map = {'Alef': ')', 'Ayin': '(', 'Bet': 'b', 'Dalet': 'd', 'Gimel': 'g', 'He': 'x', 'Het': 'h',
                'Kaf': 'k', 'Kaf-final': '\\', 'Lamed': 'l', 'Mem': '{', 'Mem-medial': 'm', 'Nun-final': '}',
                'Nun-medial': 'n', 'Pe': 'p', 'Pe-final': 'v', 'Qof': 'q', 'Resh': 'r', 'Samekh': 's',
                'Shin': '$', 'Taw': 't', 'Tet': '+', 'Tsadi-final': 'j', 'Tsadi-medial': 'c', 'Waw': 'w',
                'Yod': 'y', 'Zayin': 'z'}
    return labels, char_map
