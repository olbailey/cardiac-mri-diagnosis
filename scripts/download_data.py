import shutil
import os

import kagglehub

DESTINATION = "data/raw"

def main():

    os.makedirs(DESTINATION, exist_ok=True)

    # Download latest version
    path = kagglehub.dataset_download("samdazel/automated-cardiac-diagnosis-challenge-miccai17")
    cache_path = os.path.join(path, "database/")

    print("Path to dataset files:", cache_path)


    for item in os.listdir(cache_path):
        source = os.path.join(cache_path, item)
        dest = os.path.join(DESTINATION, item)
        shutil.move(source, dest)

    shutil.rmtree(path)

    print("Dataset moved to:", DESTINATION)

if __name__ == "__main__":
    main()
