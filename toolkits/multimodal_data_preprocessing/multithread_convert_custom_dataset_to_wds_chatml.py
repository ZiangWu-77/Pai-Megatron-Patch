import io
import json
import os
import pickle
from argparse import ArgumentParser
from typing import Any, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import webdataset as wds
import yaml
from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseWebdatasetFactory
from PIL import Image
from tqdm import tqdm
from webdataset.writer import add_handlers, default_handlers


def process_image_to_jpeg_bytes(image_path: str) -> Optional[bytes]:
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
        with Image.open(image_path) as img:
            # Check if the image is already in the desired format (JPEG and RGB)
            is_standard_jpeg = (img.format == 'JPEG' and img.mode == 'RGB')

            if is_standard_jpeg:
                # Fast path: If it's already a standard RGB JPEG, read raw bytes.
                with open(image_path, 'rb') as f:
                    return f.read()
            else:
                # Conversion path: Convert other formats to RGB.
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save the converted image to an in-memory byte buffer.
                with io.BytesIO() as buffer:
                    img.save(buffer, format="JPEG", quality=100)
                    return buffer.getvalue()

    except FileNotFoundError:
        print(f"Warning: Image file not found, skipping: {image_path}")
        return None
    except Exception as e:
        print(f"Warning: Failed to process image {image_path}. Error: {e}. Skipping.")
        return None

# --- NEW FUNCTION ---
# This function encapsulates the logic for processing a single JSON entry.
# It can be called concurrently by multiple threads.
def process_entry(args_tuple) -> Optional[Dict[str, Any]]:
    """
    Processes a single data entry from the JSON file.

    This function handles image and video frame loading and conversion for one entry.
    It's designed to be run in a separate thread/process.

    Args:
        args_tuple: A tuple containing (entry, idx, dataset_dir, sort_function).

    Returns:
        A sample dictionary ready for webdataset, or None if processing fails.
    """
    entry, idx, dataset_dir, sort_function = args_tuple
    try:
        # Process images
        image_datas = []
        image_list = entry.get("image", [])
        if isinstance(image_list, str):
            image_list = [image_list]

        for image_filename in image_list:
            full_path = os.path.join(dataset_dir, image_filename)
            image_bytes = process_image_to_jpeg_bytes(full_path)
            if image_bytes:
                image_datas.append(image_bytes)

        # Process videos
        video_datas = []
        second_per_grid_ts = []
        for video in entry.get('videos', []):
            video_noext, _ = os.path.splitext(video)
            frame_folder = os.path.join(dataset_dir, video_noext)

            fps = 2.0  # Default FPS
            if os.path.exists(frame_folder + '.json'):
                with open(frame_folder + '.json', 'r') as f:
                    fps = float(json.load(f).get('fps', 2.0))

            frames = []
            if os.path.isdir(frame_folder):
                for frame_filename in sort_function(os.listdir(frame_folder)):
                    frame_path = os.path.join(frame_folder, frame_filename)
                    frame_bytes = process_image_to_jpeg_bytes(frame_path)
                    if frame_bytes:
                        frames.append(frame_bytes)

            if len(frames) % 2 == 1:
                frames = frames[:-1]
            
            if frames: # Only add if frames were successfully processed
                video_datas.append(frames)
                second_per_grid_ts.append(1 / fps)

        # Assemble the sample dictionary
        sample = {
            "__key__": str(idx),
            "jpgs": image_datas,
            'videos': video_datas,
            "json": json.dumps({
                'conversations': entry['conversations'],
                'second_per_grid_ts': second_per_grid_ts
            }).encode("utf-8"),
        }
        return sample

    except Exception as e:
        entry_id = entry.get('id', f"index {idx}")
        print(f"\nWarning: Failed to process entry '{entry_id}'. Error: {e}. Skipping this entry.")
        return None

# --- MODIFIED FUNCTION ---
# The convert function now uses a ThreadPoolExecutor to parallelize processing.
def convert(dataset_dir, json_name, sort_function=sorted, max_count=10000, num_workers=8):
    """
    Convert a dataset to WebDataset format in parallel, ensuring all images are 3-channel JPEGs.
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

    # Prepare arguments for each task
    tasks = [(entry, idx, dataset_dir, sort_function) for idx, entry in enumerate(data)]

    with wds.ShardWriter(os.path.join(output, 'pretrain-%d.tar'), maxcount=max_count) as shard_writer:
        # Create a thread pool to process entries in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Use executor.map to apply process_entry to each task
            # Wrap with tqdm to show progress
            results = tqdm(executor.map(process_entry, tasks), total=len(tasks))
            
            # The main thread iterates through the processed results and writes them.
            # This ensures that writing to the shard_writer is sequential and thread-safe.
            for sample in results:
                if sample: # Only write if processing was successful
                    shard_writer.write(sample)

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
    argparser.add_argument('--max-samples-per-tar', default=10000, type=int)
    # --- NEW ARGUMENT ---
    # Add an argument to control the number of worker threads.
    # Default to the number of CPU cores if not specified.
    argparser.add_argument('--num-workers', default=os.cpu_count(), type=int, help="Number of worker threads for processing.")
    argparser.add_argument('--train-split', default=9, type=float)
    argparser.add_argument('--val-split', default=1, type=float)
    argparser.add_argument('--test-split', default=0, type=float)
    args = argparser.parse_args()


    output_dir = convert(
        args.dataset_root, 
        args.json, 
        max_count=args.max_samples_per_tar, 
        num_workers=args.num_workers
    )
    print(f"Generating Configurations")
    # NOTE: split_ratio: train/val/test
    split=[args.train_split, args.val_split, args.test_split]
    generate_configs(EPath(output_dir), split, num_workers=args.num_workers)
    print(f"Configurations Generated")