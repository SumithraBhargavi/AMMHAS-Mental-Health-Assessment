import h5py
import numpy as np
import os

MOSEI_PATH = r"backend/data/raw/mosei/"

def explore_csd(file_path):
    print("\n📂 Reading:", file_path)
    with h5py.File(file_path, "r") as f:
        print("Top-level keys:", list(f.keys()))
        print("Structure inside first key:")
        first_key = list(f.keys())[0]
        print(list(f[first_key].keys()))
        
        print("\nData keys:")
        print(list(f[first_key]["data"].keys()))
        
        print("\nMetadata keys:")
        print(list(f[first_key]["metadata"].keys()))

if __name__ == "__main__":
    for file in os.listdir(MOSEI_PATH):
        if file.endswith(".csd"):
            explore_csd(os.path.join(MOSEI_PATH, file))
