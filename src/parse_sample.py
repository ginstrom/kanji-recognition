import struct
import numpy as np
from PIL import Image
import os
from jis_unicode_map import jis_to_unicode

ETL9G_RECORD_SIZE = 8199

# Save output to a folder for viewing in browser
output_dir = 'etl9g_images'
os.makedirs(output_dir, exist_ok=True)

def jis2unicode(jis_code):
    return jis_to_unicode.get(jis_code, 'UNK')


def extract_etl9g_images(file_path, limit=20):
    with open(file_path, 'rb') as f:
        record_idx = 0
        while record_idx < limit:
            s = f.read(ETL9G_RECORD_SIZE)
            print(f"Record length: {len(s)}")

            if not s or len(s) != ETL9G_RECORD_SIZE:
                print(f"Skipping invalid record at index {record_idx}, got {len(s)} bytes")
                continue

            try:
                # Read metadata
                r = struct.unpack('>HH6xH6x4s4x', s[:26])
                print('r: ', r)
                jis_code = r[1]

                # Extract the last 8128 bytes as image data
                img_data = s[-8128:]
                if len(img_data) != 8128:
                    print(f"Skipping image at index {record_idx}: expected 8128 bytes, got {len(img_data)}")
                    continue

                raw = np.frombuffer(img_data, dtype=np.uint8)
                pixels = np.zeros((127, 128), dtype=np.uint8)

                for j in range(8128):
                    byte = raw[j]
                    hi = (byte >> 4) * 17
                    lo = (byte & 0x0F) * 17
                    y = (j * 2) // 128
                    x = (j * 2) % 128
                    pixels[y, x] = hi
                    pixels[y, x + 1] = lo

                img = Image.fromarray(pixels, mode='L')
                unicode_char = jis2unicode(jis_code)
                output_path = f"{output_dir}/{record_idx:05}_{unicode_char}.png"
                print("output_path: ", output_path)
                img.save(output_path)

                record_idx += 1
            except Exception as e:
                print(f"Error processing record {record_idx}: {e}")
                continue


# Example usage (adjust path and limit as needed):
extract_etl9g_images('/data/ETL9G/ETL9G_01', limit=200)