from WaveletImageCoder import WaveletImageEncoder
import argparse
from PIL import Image
import numpy as np
from utils import psnr, comp_ratio, bpp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress a file to the ZTC format")
    parser.add_argument('file', type=str, help="File to compress")
    parser.add_argument('--output', type=str, default="output.ztc", help="Output filename")
    parser.add_argument("--max-passes", type=float, default=float('inf'), help="Maximum number of Image Packets")
    args = parser.parse_args()

    print("Compressing Image")

    img = np.array(Image.open(args.file))
    encoder = WaveletImageEncoder(args.max_passes)
    encoder.encode(img, args.output)

    print(f"Saved output to {args.output}")
    print(f"BPP: {bpp(args.output)}")
    print(f"Compression Ratio: {comp_ratio(args.file, args.output)}")
