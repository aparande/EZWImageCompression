from WaveletImageCoder import WaveletImageDecoder
import argparse
from PIL import Image
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decompress a file from the ZTC format")
    parser.add_argument('file', type=str, help="File to decompress")
    parser.add_argument('--output', type=str, default="output.png", help="Output filename")
    args = parser.parse_args()

    decoder = WaveletImageDecoder()
    image = decoder.decode(args.file)
    Image.fromarray(image).save(args.output)
