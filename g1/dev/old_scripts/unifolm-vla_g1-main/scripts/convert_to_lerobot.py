import os
import cv2
import h5py
import numpy as np
import pandas as pd

# ----------------------------
# CONFIG
# ----------------------------
SOURCE_DIR = "pybullet_data"
TARGET_DIR = "lerobot_dataset"
DATASET_NAME = "g1_pick_task"

VIDEO_DIR = os.path.join(TARGET_DIR, "videos", DATASET_NAME)
TRANS_DIR = os.path.join(TARGET_DIR, "transitions", DATASET_NAME)

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(TRANS_DIR, exist_ok=True)

csv_rows = []

# ----------------------------
# HELPER: CREATE VIDEO
# ----------------------------
def create_video(image_paths, output_path):
    frame = cv2.imread(image_paths[0])
    h, w, _ = frame.shape

    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          15,
                          (w, h))

    for img_path in image_paths:
        img = cv2.imread(img_path)
        out.write(img)

    out.release()


# ----------------------------
# MAIN LOOP
# ----------------------------
episodes = sorted(os.listdir(SOURCE_DIR))

for idx, ep in enumerate(episodes):
    ep_path = os.path.join(SOURCE_DIR, ep)

    # Load images
    images = sorted([f for f in os.listdir(ep_path) if "rgb" in f])
    image_paths = [os.path.join(ep_path, img) for img in images]

    # Create video
    video_path = os.path.join(VIDEO_DIR, f"{idx}.mp4")
    create_video(image_paths, video_path)

    # Load actions
    actions = np.load(os.path.join(ep_path, "actions.npy"))

    # Optional: load states
    states_path = os.path.join(ep_path, "states.npy")
    states = np.load(states_path) if os.path.exists(states_path) else None

    # Save HDF5
    h5_path = os.path.join(TRANS_DIR, f"{idx}.h5")

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("actions", data=actions)

        if states is not None:
            f.create_dataset("states", data=states)

    # CSV metadata
    csv_rows.append({
        "episode_id": idx,
        "video_path": video_path,
        "transition_path": h5_path,
        "language_instruction": "pick the object"
    })


# ----------------------------
# SAVE CSV
# ----------------------------
csv_path = os.path.join(TARGET_DIR, f"{DATASET_NAME}.csv")
df = pd.DataFrame(csv_rows)
df.to_csv(csv_path, index=False)

print("✅ Conversion complete!")
