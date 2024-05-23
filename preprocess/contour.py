'''Python version 3.8.18 '''

import os
import cv2

def init_contour_finding(img_dir, epidural, intraparenchymal,
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

def calculate_gray_percentage(gray_image, threshold=128):
    gray_image = np.uint8(gray_image)
    # Threshold the grayscale image
    _, binary_mask = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

    # Count the number of white pixels (gray) in the binary mask
    gray_pixel_count = np.sum(binary_mask == 255)

    # Calculate the total number of pixels in the image
    total_pixels = gray_image.size

    # Calculate the percentage of gray pixels
    gray_percentage = (gray_pixel_count / total_pixels) * 100

    return gray_percentage

def count_size(gray_img, contour):
    # outside contour to be black
    mask = np.zeros_like(gray_img, dtype=np.uint8) 
    cv2.drawContours(mask, [contour], 0, 255, thickness=cv2.FILLED)
    result = cv2.bitwise_and(gray_img, gray_img, mask=mask)

    return np.count_nonzero(result!=0)

def find_contour(gray_image, img_name):
    # Convert normalized grayscale image to uint8
    gray_image = gray_image.astype(np.uint8)
    
    threshold = 55
    
    # Threshold the grayscale image
    _, thresh = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on bounding box size
    img_size = gray_image.shape[1]

    brightest_contour = None
    max_brightness = 0

    # Iterate through filtered contours
    for contour in contours:
        # Filter out contours with fewer than 5 points
        # Calculate the average brightness inside the contour
        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask, [contour], 0, 255, thickness=cv2.FILLED)
        average_brightness = np.mean(gray_image[mask == 255])

        # First contour or not?
        if brightest_contour is not None:
            size_now = count_size(gray_img, brightest_contour)
            size_new = count_size(gray_img, contour)

            if (size_new > size_now):
                max_brightness = average_brightness
                brightest_contour = contour

        else:
            max_brightness = average_brightness
            brightest_contour = contour

    return brightest_contour