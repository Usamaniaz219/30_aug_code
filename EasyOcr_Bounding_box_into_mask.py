import cv2
import numpy as np
import math
import easyocr
import os 
import cv2



def load_image(image_path):
    return cv2.imread(image_path)

def calculate_num_rows_and_cols(image, tile_width, tile_height):
    num_rows = math.ceil(image.shape[0] / tile_height)
    num_cols = math.ceil(image.shape[1] / tile_width)
    return num_rows, num_cols

def extract_tile(image, start_x, start_y, tile_width, tile_height):
    end_x = min(start_x + tile_width, image.shape[1])
    end_y = min(start_y + tile_height, image.shape[0])
    return image[start_y:end_y, start_x:end_x]

def create_mask_for_bboxes(image_shape, bounding_boxes):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    print("mask shape:",mask)
    for box in bounding_boxes:
        box = np.array(box, dtype=np.int32)
        box = box.reshape((-1, 1, 2))

# Draw the contours on the mask
        # cv2.drawContours(mask, [box], contourIdx=-1, color=255, thickness=2)

        cv2.fillPoly(mask, [box], 255)
    return mask

def detect_text_in_tile(image, tile_width, tile_height, reader):
    bounding_boxes = []
    num_rows, num_cols = calculate_num_rows_and_cols(image, tile_width, tile_height)

    for r in range(num_rows):
        for c in range(num_cols):
            start_x = c * tile_width
            start_y = r * tile_height
            tile = extract_tile(image, start_x, start_y, tile_width, tile_height)

            # result = reader.readtext(tile,text_threshold=0.7,low_text=0.6,link_threshold=0.4)
            # result = reader.readtext(tile,text_threshold=0.7,low_text=0.55,slope_ths=0.001)
            # result = reader.readtext(tile, text_threshold=0.3,min_size = 5, low_text=0.55,adjust_contrast=0.8,mag_ratio=1.5)
            # result = reader.readtext(tile, text_threshold=0.1,min_size = 10, low_text=0.52,mag_ratio=1.5) 
            result = reader.readtext(tile) 

            if len(result) > 0:
                for bbox, text, _ in result:
                    bbox = np.array(bbox, dtype=np.float32)
                    rect = cv2.minAreaRect(bbox)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    box[:, 0] += start_x
                    box[:, 1] += start_y
                    bounding_boxes.append(box.tolist())

    return bounding_boxes

def main(image, tile_width, tile_height):
    # image = load_image(image_path)
    reader = easyocr.Reader(['en'], gpu=True)

    bounding_boxes = detect_text_in_tile(image, tile_width, tile_height, reader)
    # print("Bounding Boxes",bounding_boxes)
    mask = create_mask_for_bboxes(image.shape, bounding_boxes)

    return bounding_boxes, mask

