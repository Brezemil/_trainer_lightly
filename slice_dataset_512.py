import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image

SRC_DIR = r"C:\Users\emil_brezovsky\Documents\GitHub\_dataset_unlabeled"
DST_DIR = r"C:\Users\emil_brezovsky\Documents\GitHub\_dataset_unlabeled_512"
TILE_SIZE = 512
QUALITY = 95

# 4 Quadrant crops for 1024x1024 images
CROPS = [
    (0, 0, 512, 512),
    (512, 0, 1024, 512),
    (0, 512, 512, 1024),
    (512, 512, 1024, 1024),
]


def process_image(filename: str) -> int:
    """Slice a single image into 4 sub-tiles of 512x512."""
    src_path = os.path.join(SRC_DIR, filename)
    base_name, ext = os.path.splitext(filename)
    tiles_created = 0

    try:
        with Image.open(src_path) as img:
            w, h = img.size
            if w == 1024 and h == 1024:
                for idx, crop_box in enumerate(CROPS):
                    dst_filename = f"{base_name}_t{idx}{ext}"
                    dst_path = os.path.join(DST_DIR, dst_filename)
                    if not os.path.exists(dst_path):
                        tile = img.crop(crop_box)
                        tile.save(dst_path, quality=QUALITY)
                    tiles_created += 1
            else:
                # Handle generic image sizes
                for x_idx, x in enumerate(range(0, w, TILE_SIZE)):
                    for y_idx, y in enumerate(range(0, h, TILE_SIZE)):
                        right = min(x + TILE_SIZE, w)
                        bottom = min(y + TILE_SIZE, h)
                        if (right - x) >= 256 and (bottom - y) >= 256:
                            dst_filename = f"{base_name}_tile_{x_idx}_{y_idx}{ext}"
                            dst_path = os.path.join(DST_DIR, dst_filename)
                            if not os.path.exists(dst_path):
                                tile = img.crop((x, y, right, bottom))
                                if tile.size != (TILE_SIZE, TILE_SIZE):
                                    tile = tile.resize(
                                        (TILE_SIZE, TILE_SIZE),
                                        Image.Resampling.BILINEAR,
                                    )
                                tile.save(dst_path, quality=QUALITY)
                            tiles_created += 1
    except Exception as e:
        print(f"Error processing {filename}: {e}", file=sys.stderr)

    return tiles_created


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    files = [
        f
        for f in os.listdir(SRC_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]
    total_files = len(files)
    print("=" * 80)
    print(" SLICING DATASET TO 512x512")
    print("=" * 80)
    print(f" Source Directory:      {SRC_DIR}")
    print(f" Destination Directory: {DST_DIR}")
    print(f" Total Source Images:   {total_files}")
    print(f" Target Sub-tile Size:  {TILE_SIZE}x{TILE_SIZE} px")
    print("=" * 80 + "\n")

    num_workers = min(os.cpu_count() or 4, 16)
    print(f" Launching multi-core slicing across {num_workers} processes...")

    start_time = time.time()
    completed_count = 0
    total_tiles = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_image, fname): fname for fname in files}

        for future in as_completed(futures):
            completed_count += 1
            tiles = future.result()
            total_tiles += tiles

            if completed_count % 5000 == 0 or completed_count == total_files:
                elapsed = time.time() - start_time
                fps = completed_count / elapsed if elapsed > 0 else 0
                remaining_sec = (total_files - completed_count) / fps if fps > 0 else 0
                print(
                    f" [{completed_count:,}/{total_files:,}] images sliced "
                    f"({total_tiles:,} tiles) | Speed: {fps:.1f} img/s | "
                    f"Elapsed: {elapsed / 60:.1f}m | Remaining: {remaining_sec / 60:.1f}m"
                )

    total_elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(" SLICING COMPLETE!")
    print(f" Total Sub-tiles Created: {total_tiles:,}")
    print(f" Output Location:        {DST_DIR}")
    print(f" Total Execution Time:   {total_elapsed / 60:.2f} minutes")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
