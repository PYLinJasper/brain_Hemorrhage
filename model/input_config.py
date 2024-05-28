import os
import ast
from random import sample

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
            cv2.polylines(background, [vertices_scaled], isClosed=True, color=(255, 255, 255), thickness=3)

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