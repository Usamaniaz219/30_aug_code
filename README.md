Here's an updated `README.md` for your new code:

---

# Text Erasure from Map Images Using Contours and Bounding Boxes

This project implements a technique to erase text from map images using bounding box masks, contours, and morphological operations. It involves detecting the text regions within specific zones and removing them while retaining important features in the images. The processed images are saved with both the text and zone masks.

## Project Overview

- **Input:** A set of source images and their corresponding bounding box masks.
- **Output:** Processed images where the text has been erased from specified zones based on contours.

### Main steps include:
1. **Bounding box extraction** using EasyOCR for text detection.
2. **Contour-based text erasure** by identifying which text lies within specific zones.
3. **Dilation operations** applied depending on the resolution of the image to refine the results.

## Dependencies

- Python 3.x
- OpenCV
- NumPy
- EasyOCR
- Shapely

### Install dependencies using pip:

```bash
pip install opencv-python numpy shapely easyocr
```

## Directory Structure

```
project_root/
│
├── A/                      # Input source images folder
│    ├── image1.jpg
│    └── image2.jpg
│
├── B/                      # Input bounding box mask images folder
│    ├── image1/
│    │   ├── mask1.jpg
│    │   └── mask2.jpg
│    └── image2/
│        ├── mask1.jpg
│        └── mask2.jpg
│
└── output/                 # Output folder for processed images
     ├── image1_output_mask.jpg
     ├── image1_text_mask.jpg
     └── ...
```

## How to Use

1. **Prepare the input data:**
   - Add the original map images in the folder defined as `input__image_directory`.
   - Add corresponding bounding box mask images in the folder defined as `mask_image_dir`. Ensure that each image's bounding box masks are placed inside a subdirectory matching the image name.

2. **Run the script:**

```bash
python3 text_erasing_from_mask_image_30_updated_aug_2024.py
```

3. **Output:**
   - The processed images will be saved in the `output_directory`, with filenames indicating the image and text masks.

## Key Functions

### `retain_contours()`
This function extracts contours from the input bounding boxes and applies them to the grayscale version of the source image. It returns the masked output where text contours are applied to the thresholded map image.

### `text_erasing_using_pointpolygon_text()`
This function erases text by evaluating the intersection of text contours within the zones defined by the mask image. It calculates the percentage of intersection and retains text contours that fall inside the zone contours.

### `apply_dilation()`
This function applies dilation to the text masks based on the image resolution. Different iterations of dilation are applied depending on the resolution range of the input map image, ensuring that the text erasure is smooth and robust.

### `process_images()`
This is the main function that processes the images in bulk. It iterates through all the images in the input directory, applies the mask and contour logic, and saves the output in the specified folder.

## Customization

- **Bounding Box Extraction:**
   The bounding boxes are obtained using the EasyOCR library, which extracts text from images. You can adjust the bounding box extraction method in `EasyOcr_Bounding_box_into_mask.py` if needed.
  
- **Thresholding and Dilation Parameters:**
   You can modify the kernel sizes and iterations in the `apply_dilation()` function to fine-tune the text erasure process, especially if working with images of varying resolutions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

### Example Usage

For different resolution ranges, the `apply_dilation()` function handles the dilation iterations:

- For images between 3000x5000 pixels, 1 iteration of dilation is applied.
- For images larger than 5000 pixels, multiple iterations of dilation are applied to improve text erasure results.

To modify the resolution-based logic, adjust the conditions in the function:

```python
if resolution >= 3000 and resolution <= 5000:
    iterations = 1
```

