import shutil
import os
import requests

import kagglehub

DESTINATION = "data/raw"

def main():

    os.makedirs(DESTINATION, exist_ok=True)

    # Download latest version
    try:
        path = kagglehub.dataset_download("samdazel/automated-cardiac-diagnosis-challenge-miccai17")
    except requests.exceptions.SSLError as e:
        print("SSL verification is being blocked please follow the guidance in the README.md found in the data directory")
        print("https://www.kaggle.com/datasets/samdazel/automated-cardiac-diagnosis-challenge-miccai17")

        return

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
