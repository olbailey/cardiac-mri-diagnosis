import os

import SimpleITK as sitk

def resize_image(itk_image: sitk.Image, voxel_spacing: float, is_mask=False) -> sitk.Image:
    """
    Resamples the image to set the voxel size equal to the mean so that the images represent the anatomy uniformly.

    Args:
        itk_image (sitk.Image): The sitk image volume to be processed.
        target_spacing (float): mean voxel spacing value.
        is_mask (bool, optional): _description_. Defaults to False.

    Returns:
        sitk.Image: _description_
    """
    # target_spacing = (voxel size x, voxel size y, dimensions)
    original_spacing = itk_image.GetSpacing()   # (x, y, z) mm
    original_size = itk_image.GetSize()         # (W, H, D) pixels
    target_spacing = (voxel_spacing, voxel_spacing, original_spacing[2])

    # Compute new pixel dimensions so physical size is preserved
    new_size = [
        int(round(origin_size * origin_space / target_space))
        for origin_size, origin_space, target_space in zip(original_size, original_spacing, target_spacing)
    ]

    resampler = sitk.ResampleImageFilter()

    # Tells the filter what voxel spacing the output image should have
    resampler.SetOutputSpacing(target_spacing)

    # Tells the filter how many voxels the output image should have, in each dimension
    resampler.SetSize(new_size)

    # You want the output to have the same orientation as the input
    # Only changing sampling density, not rotating or flipping the anatomy.
    resampler.SetOutputDirection(itk_image.GetDirection())

    # This anchors the output grid to start at the same physical point as the input, 
    # so the resampled image still lines up spatially with the original
    resampler.SetOutputOrigin(itk_image.GetOrigin())

    # identity transform — no rotation/translation
    resampler.SetTransform(sitk.Transform())

    # fill value for any empty regions
    resampler.SetDefaultPixelValue(0)                 

    if is_mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        # smooth interpolation for intensity images
        resampler.SetInterpolator(sitk.sitkLinear)   

    return resampler.Execute(itk_image)

def get_patient_frame(patient_path: str, finding_lowest=True) -> int:
    """Used to get the targetted frame digit from the patient for processing

    Args:
        patient_path (str): path to the patient folder in the raw dataset
        finding_lowest (bool, optional): for whether to search for the lowest or largest frame. Defaults to True.

    Returns:
        int: The digit for the lowest or largest frame from the patient folder
    """
    frame_digits = []
    for file_name in os.listdir(patient_path):
        if "frame" in file_name:
            # frame file with be structured for example patient001_frame01...
            # so split to get 01... then retrieve the first two characters to get the number
            frame_digit = file_name.split("frame")[1][:2]
            frame_digits.append(frame_digit)

    if finding_lowest:
        lowest_frame_digit = min(list(map(lambda x: int(x), frame_digits))) 
    else:
        lowest_frame_digit = max(list(map(lambda x: int(x), frame_digits))) 

    
    for digit in frame_digits:
        if f"{lowest_frame_digit:02d}" == digit:
            return digit

def get_patient_label(patient_dir_path: str) -> str:
    """Retrieving the patient diagnosis label

    Args:
        patient_dir_path (str): _description_

    Raises:
        RuntimeError: _description_

    Returns:
        str: _description_
    """
    config_path = os.path.join(patient_dir_path, "Info.cfg")
    with open(config_path, 'r') as config_file:
        for line in config_file.readlines():
            if "Group" in line:
                # line including the group will be structured as "Group: type"
                return line.strip().split(": ")[1]

        raise RuntimeError("Error! 'Group' config not found")