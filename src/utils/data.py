import os

def get_patient_frame(patient_path, finding_lowest=True):
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

def get_patient_label(patient_dir_path):
    config_path = os.path.join(patient_dir_path, "Info.cfg")
    with open(config_path, 'r') as config_file:
        for line in config_file.readlines():
            if "Group" in line:
                # line including the group will be structured as "Group: type"
                return line.strip().split(": ")[1]

        raise RuntimeError("Error! 'Group' config not found")