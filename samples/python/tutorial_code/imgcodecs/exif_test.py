import cv2 as cv
import numpy as np

def dump_exif(exif_entries, title="EXIF DUMP"):
    """
    Print all EXIF entries in a readable format.
    """
    print(f"=== {title} ===")
    for block in exif_entries:
        print('----------------------')
        for entry in block:
            print(entry.dumpAsString())

def get_exif_block(metadata_types, metadata):
    """
    Find the EXIF metadata block from metadata.
    """
    for i, t in enumerate(metadata_types):
        if t == cv.IMAGE_METADATA_EXIF:
            return metadata[i][0]
    return None

def print_exif(path):
    # Load image and metadata
    img, metadata_types, metadata = cv.imreadWithMetadata(path)
    if img is None:
        print("Failed to load image:", path)
        return

    # Extract the raw EXIF block
    raw_exif = get_exif_block(metadata_types, metadata)
    if raw_exif is None:
        print("No EXIF metadata found")
        return

    # Decode EXIF into structured entries
    result, exif_entries = cv.decodeExif(raw_exif)
    if not result:
        print("Parsing EXIF raw data failed")
        return

    # Dump original
    dump_exif(exif_entries, title="ORIGINAL EXIF")

    # Encode with original metadata
    _, buf = cv.imencodeWithMetadata(".jpg", img, metadata_types, metadata)

    # Re-encode EXIF
    res, raw_exif_data = cv.encodeExif(exif_entries)
    if not res:
        print("Failed to encode EXIF")
        return

    new_metadata = [np.frombuffer(raw_exif_data, dtype=np.uint8), metadata[1], metadata[2]]

    _, buf = cv.imencodeWithMetadata(".jpg", img, metadata_types, new_metadata)

    # Reload and verify
    _, metadata_types2, metadata2 = cv.imdecodeWithMetadata(buf)
    raw_exif2 = get_exif_block(metadata_types2, metadata2)

    _, exif_entries2 = cv.decodeExif(raw_exif2)
    print("*" * 60)
    dump_exif(exif_entries2, title="RELOADED EXIF")

if __name__ == "__main__":
    print_exif(cv.samples.findFile("board.jpg"))
