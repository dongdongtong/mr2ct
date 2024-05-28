import cv2
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from preprocess.brain_extractor import BrainExtractor


def apply_otsu_to_each_slice(input_image_path, output_image_path):
    # Read the input CT image
    input_image = sitk.ReadImage(input_image_path)
    
    # Convert the SimpleITK image to a numpy array
    input_array = sitk.GetArrayFromImage(input_image)

    if np.max(input_array) < 0:
        pass
    else:
        input_array = np.clip(input_array, -1024, 3071)

    # initialize the brain extractor
    be = BrainExtractor()
    
    # Initialize an empty list to store the processed slices
    processed_slices = []
    
    # Iterate over each axial slice
    for slice_index in tqdm(range(input_image.GetDepth())):
        # Get the current axial slice
        slice_array = input_array[slice_index, :,:]
        
        # Convert the slice to uint8 (required by OpenCV)
        slice_uint8 = cv2.normalize(slice_array, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Apply Otsu's thresholding method to the slice
        _, binary_slice = cv2.threshold(slice_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        binary_slice = be._fill_hole(binary_slice[:, :, None])

        cv2.erode(binary_slice, np.ones((5, 5), np.uint8), binary_slice, iterations=2)
        
        # Append the processed slice to the list
        processed_slices.append(binary_slice)
    
    # Convert the list of processed slices to a numpy array
    processed_array = np.array(processed_slices)
    
    # Convert the numpy array back to a SimpleITK image
    processed_image = be.array2image(processed_array, input_image)

    # mask the CT image
    processed_image = sitk.Mask(input_image, processed_image)
    
    # Write the processed image to disk
    sitk.WriteImage(processed_image, output_image_path)
    
    print("Otsu thresholding applied to each axial slice. Resulting image saved as:", output_image_path)

# Example usage
input_image_path = "data/pipeline_niigz_betneck/ct/BICHENQIONG_0000.nii.gz"  # Replace with the path to your input CT image
output_image_path = "./BICHENQIONG_0000_otsuonslice_erode_2.nii.gz"  # Specify the path for the output segmented image

apply_otsu_to_each_slice(input_image_path, output_image_path)