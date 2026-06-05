import os
import numpy as np
import h5py

# ------------------  PATHS (EDIT IF NEEDED)  ------------------ #

DATASET_ROOT = r"C:\Users\ragha\Downloads\archive\CMU-MOSEI"
OUTPUT_ROOT = r"D:\Software\npy"   # you wanted folder named 'npy'

os.makedirs(OUTPUT_ROOT, exist_ok=True)


# ------------------  CORE HELPER  ------------------ #

def save_sequence(csd_rel_path, out_subdir, dataset_key="features", root_key=None):
    """
    Generic converter:
    - opens a .csd file with h5py
    - goes into <root_key>/data
    - for each video id, picks a dataset (e.g. 'features')
    - saves it as <video_id>.npy in out_subdir
    """

    csd_path = os.path.join(DATASET_ROOT, csd_rel_path)
    out_dir = os.path.join(OUTPUT_ROOT, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== Processing {csd_path} -> {out_dir} ===")

    with h5py.File(csd_path, "r") as f:
        # If you don't know the root group name, take the first one (like 'glove_vectors', 'COVAREP', etc.)
        if root_key is None:
            root_key = list(f.keys())[0]

        root = f[root_key]
        data_group = root["data"]

        print(f"Root key: {root_key} | #items in data: {len(data_group)}")

        for vid in data_group.keys():
            node = data_group[vid]

            # Case 1: it's a group (usual for MOSEI)
            if isinstance(node, h5py.Group):
                if dataset_key in node:
                    ds = node[dataset_key]
                else:
                    # fallback: use the first dataset inside the group
                    ds_names = [k for k in node.keys() if isinstance(node[k], h5py.Dataset)]
                    if not ds_names:
                        print(f"  [WARN] No dataset found for {vid}, skipping")
                        continue
                    ds = node[ds_names[0]]

            # Case 2: it's directly a dataset
            elif isinstance(node, h5py.Dataset):
                ds = node
            else:
                print(f"  [WARN] Unknown type for {vid}, skipping")
                continue

            arr = ds[()]  # read as numpy array
            out_path = os.path.join(out_dir, f"{vid}.npy")
            np.save(out_path, arr)

        print(f"Done: saved {len(data_group)} items to {out_dir}")


# ------------------  MAIN: HANDLE ALL YOUR CSD FILES  ------------------ #

def main():
    # 1) Visual Facet 42
    save_sequence(
        csd_rel_path=r"visuals\CMU_MOSEI_VisualFacet42.csd",
        out_subdir="visual_facet",
        dataset_key="features",   # numeric visual features
    )

    # 2) Visual OpenFace2
    save_sequence(
        csd_rel_path=r"visuals\CMU_MOSEI_VisualOpenFace2.csd",
        out_subdir="visual_openface",
        dataset_key="features",
    )

    # 3) Audio COVAREP
    save_sequence(
        csd_rel_path=r"acoustics\CMU_MOSEI_COVAREP.csd",
        out_subdir="audio_covarep",
        dataset_key="features",
    )

    # 4) Timestamped Word Vectors (GloVe text embeddings)
    save_sequence(
        csd_rel_path=r"languages\CMU_MOSEI_TimestampedWordVectors.csd",
        out_subdir="text_wordvec",
        dataset_key="features",
    )

    # 5) Timestamped Words (actual tokens; may not be numeric, but we still save)
    save_sequence(
        csd_rel_path=r"languages\CMU_MOSEI_TimestampedWords.csd",
        out_subdir="text_words",
        dataset_key="words",   # if 'words' doesn't exist, code will fall back to first dataset
    )

    # 6) Timestamped Phones
    save_sequence(
        csd_rel_path=r"languages\CMU_MOSEI_TimestampedPhones.csd",
        out_subdir="text_phones",
        dataset_key="phones",  # again, will fall back if not present
    )

    # 7) Labels (sentiment/emotion labels)
    save_sequence(
        csd_rel_path=r"labels\CMU_MOSEI_Labels.csd",
        out_subdir="labels",
        dataset_key="features",   # labels are usually stored under 'features'
    )

    print("\n✔ All .csd files processed.")


if __name__ == "__main__":
    main()
