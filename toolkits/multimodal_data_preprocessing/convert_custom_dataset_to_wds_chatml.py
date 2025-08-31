import io
import json
import os
import pickle
from argparse import ArgumentParser
from typing import Any

import cv2
import numpy as np
import webdataset as wds
import yaml
from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseWebdatasetFactory
from PIL import Image
from tqdm import tqdm
from webdataset.writer import add_handlers, default_handlers


def process_image_to_jpeg_bytes(image_path: str) -> bytes | None:
    """
    Ensures an image is a 3-channel RGB JPEG and returns its bytes.

    - If the input is already a 3-channel JPEG, its original bytes are read and returned directly to avoid re-compression.
    - If the input is a PNG, grayscale image, RGBA, or any other format, it is opened,
      converted to 3-channel RGB, and then encoded as a high-quality JPEG.

    Args:
        image_path: The file path to the image.

    Returns:
        The image data as JPEG bytes, or None if an error occurs.
    """
    try:
        # Open the image using Pillow to inspect it without fully decoding if not needed
        img = Image.open(image_path)

        # Check if the image is already in the desired format (JPEG and RGB)
        # 'img.format' tells us the original file format. 'img.mode' tells us the color mode.
        is_standard_jpeg = (img.format == 'JPEG' and img.mode == 'RGB')

        if is_standard_jpeg:
            # If it's already a standard RGB JPEG, we can take the fast path.
            # Close the PIL image handle and read the raw bytes from the file.
            img.close()
            with open(image_path, 'rb') as f:
                return f.read()
        else:
            # If it's any other format (PNG, GIF, grayscale, RGBA, etc.), we need to convert it.
            # Convert to RGB mode. This handles grayscale ('L'), RGBA ('RGBA'), etc.
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Save the converted image to an in-memory byte buffer as a high-quality JPEG.
            with io.BytesIO() as buffer:
                # quality=95 is a good balance between size and quality.
                img.save(buffer, format="JPEG", quality=100)
                return buffer.getvalue()

    except FileNotFoundError:
        print(f"Warning: Image file not found, skipping: {image_path}")
        return None
    except Exception as e:
        print(f"Warning: Failed to process image {image_path}. Error: {e}. Skipping.")
        return None

def convert(dataset_dir, json_name, sort_function=sorted, max_count=10000):
    """
    Convert a dataset to WebDataset format, ensuring all images are 3-channel JPEGs.
    Includes error handling for individual data entries.
    """
    json_file = os.path.join(dataset_dir, json_name)
    output = os.path.join(dataset_dir, 'wds')

    if not os.path.exists(output):
        os.mkdir(output)

    with open(json_file, 'r') as f:
        data = json.load(f)

    # Handlers remain simple
    add_handlers(default_handlers, "jpgs", pickle.dumps)
    add_handlers(default_handlers, "videos", pickle.dumps)

    has_idx = None
    with wds.ShardWriter(os.path.join(output, 'pretrain-%d.tar'), maxcount=max_count) as shard_writer:
        for idx, entry in enumerate(tqdm(data)):
            # --- MODIFICATION ---
            # Add a try-except block to handle errors within a single data entry.
            # This makes the script robust against malformed entries in the JSON file.
            try:
                image_datas = []
                if type(entry.get("image")) == str:
                    entry["image"] = [entry["image"]]

                for image_filename in entry.pop('image', []):
                    full_path = os.path.join(dataset_dir, image_filename)
                    image_bytes = process_image_to_jpeg_bytes(full_path)
                    if image_bytes:
                        image_datas.append(image_bytes)

                video_datas = []
                second_per_grid_ts = []

                for video in entry.pop('videos', []):
                    video_noext, _ = os.path.splitext(video)
                    frame_folder = os.path.join(dataset_dir, video_noext)
                    if os.path.exists(frame_folder + '.json'):
                        with open(frame_folder + '.json', 'r') as f:
                            fps = float(json.load(f)['fps'])
                    else:
                        fps = 2.0

                    frames = []
                    # Check if frame_folder actually exists before trying to list its directory
                    if os.path.isdir(frame_folder):
                        for frame_filename in sort_function(os.listdir(frame_folder)):
                            frame_path = os.path.join(frame_folder, frame_filename)
                            frame_bytes = process_image_to_jpeg_bytes(frame_path)
                            if frame_bytes:
                                frames.append(frame_bytes)

                    if len(frames) % 2 == 1:
                        frames = frames[:-1]
                    video_datas.append(frames)
                    second_per_grid_ts.append(1 / fps)

                if has_idx is None:
                    has_idx = 'id' in entry
                assert has_idx == ('id' in entry), "All entries should either all contain idx or not."

                sample = {
                    # "__key__": entry.pop('id', str(idx)),
                    "__key__": str(idx),
                    "jpgs": image_datas,
                    'videos': video_datas,
                    # This line is a common point of failure if 'conversations' is missing.
                    "json": json.dumps({
                        'conversations': entry['conversations'],
                        'second_per_grid_ts': second_per_grid_ts
                    }).encode("utf-8"),
                }
                shard_writer.write(sample)

            except Exception as e:
                # If any error occurs while processing an entry, print a warning and continue.
                entry_id = entry.get('id', f"index {idx}")
                print(f"\nWarning: Failed to process entry '{entry_id}'. Error: {e}. Skipping this entry.")
                print(entry, image_bytes is not None)
                continue

    print(f"Dataset successfully converted to wds")
    return output

def generate_configs(path: EPath, split, shuffle_tars=True, num_workers=32):
    path = path.absolute()
    all_tars = list(path.glob("**/*.tar")) + list(path.glob("**/*.tgz"))
    all_tars = [str(p.relative_to(path)) for p in sorted(all_tars)]
    split_parts_ratio = [("train", split[0]), ("val", split[1]), ("test", split[2])]
    split_parts_patterns = None

    # NOTE: generate .info.yaml and split.yaml
    _ = BaseWebdatasetFactory.prepare_dataset(
        path,
        all_tars,
        split_parts_ratio=split_parts_ratio,
        split_parts_patterns=split_parts_patterns,
        tar_index_only=False,
        shuffle_seed=42 if shuffle_tars else None,
        workers=num_workers,
    )

    # NOTE: dump dataset.yaml
    metadata = {
        '__class__': 'ChatMLWebdataset',
        '__module__': 'megatron_patch.data.energon.chatml',
        'field_map': {
            'imgs': 'jpgs',
            'videos': 'videos',
            'conversation': 'json'
        }
    }
    with open(os.path.join(path.url, '.nv-meta', 'dataset.yaml'), 'w') as f:
        yaml.safe_dump(metadata, f)

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--dataset-root', required=True, type=str)
    argparser.add_argument('--json', default='dataset.json', type=str)
    argparser.add_argument('--max-samples-per-tar', default=10000, type=float)
    argparser.add_argument('--train-split', default=9, type=float)
    argparser.add_argument('--val-split', default=1, type=float)
    argparser.add_argument('--test-split', default=0, type=float)
    args = argparser.parse_args()


    output_dir = convert(args.dataset_root, args.json, max_count=args.max_samples_per_tar)
    print(f"Generating Configurations")
    # NOTE: split_ratio: train/val/test
    split=[args.train_split, args.val_split, args.test_split]
    generate_configs(EPath(output_dir), split)
    print(f"Configurations Generated")
