from bitarray import bitarray
import numpy as np
import utils
from zerotree import ZeroTreeDecoder, ZeroTreeEncoder, ZeroTreeScan
import pywt

SOI_MARKER = bytes.fromhex("FFD8") # Start of Image
SOS_MARKER = bytes.fromhex("FFDA") # Start of Scan
EOI_MARKER = bytes.fromhex("FFD9") # End of Image

WAVELET = "db2"

class WaveletImageEncoder():
    def __init__(self, levels, max_passes):
        self.levels = levels
        self.max_passes = max_passes

    def encode(self, image, filename):
        M, N = image.shape[:2]

        with open(filename, 'wb') as fh:
            # Write the header
            fh.write(SOI_MARKER)
            fh.write(M.to_bytes(2, "big"))
            fh.write(N.to_bytes(2, "big"))
            fh.write(self.levels.to_bytes(2, "big"))

            image = image.astype(np.float64)

            encoders = self.build_encoders(image)
            for enc in encoders:
                fh.write(int(enc.start_thresh).to_bytes(2, 'big'))

            i = 0
            writes = float('inf')
            
            while writes != 0 and i < self.max_passes:
                writes = 0

                for enc in encoders:
                    fh.write(SOS_MARKER)
                    if i < len(enc["dominant"]):
                        enc["dominant"][i].tofile(fh)
                        writes += 1
                for enc in encoders:
                    fh.write(SOS_MARKER)
                    if i < len(enc["secondary"]):
                        enc["secondary"][i].tofile(fh)
                        writes += 1
                fh.write(SOS_MARKER)
                i += 1
            fh.write(EOI_MARKER)

    def build_encoders(self, image):
        ycbcr = utils.RGB2YCbCr(image)
        encoders = [] 
        M, N = image.shape[:2]
        for i in range(3):
            channel = ycbcr[:, :, i] if i == 0 else utils.resize(ycbcr[:, :, i], M // 2, N // 2)
            encoders.append(ZeroTreeEncoder(channel, WAVELET, self.levels if i == 0 else self.levels - 1))

        return encoders

class WaveletImageDecoder():
    def decode(self, filename):
        with open(filename, 'rb') as fh:
            soi = fh.read(2)
            if soi != SOI_MARKER:
                raise Exception("Start of Image marker not found!")
            
            M = int.from_bytes(fh.read(2), "big")
            N = int.from_bytes(fh.read(2), "big")
            levels = int.from_bytes(fh.read(2), 'big')

            thresholds = [int.from_bytes(fh.read(2), 'big') for _ in range(3)]
            decoders = self.build_decoders(M, N, levels, thresholds)

            eoi = fh.read(2)
            if eoi != SOS_MARKER:
                raise Exception("Scan's not found!")

            while eoi != EOI_MARKER:
                for dec in decoders:
                    ba = bitarray()
                    for b in iter(lambda: fh.read(2), SOS_MARKER): # iterate until next SOS marker
                        ba.frombytes(b)
                    if len(ba) != 0:
                        scan = ZeroTreeScan.from_bits(ba, True)
                        dec.process(scan)
                for dec in decoders:
                    ba = bitarray()
                    for b in iter(lambda: fh.read(2), SOS_MARKER): # iterate until next SOS marker
                        ba.frombytes(b)
                    if len(ba) != 0:
                        scan = ZeroTreeScan.from_bits(ba, False)
                        dec.process(scan)
                        
                eoi = fh.read(2)

            image = np.zeros((M, N, 3))
            for i, dec in enumerate(decoders):
                image[:, :, i] = dec.getImage() if i == 0 else utils.resize(dec.getImage(), M, N)

        return utils.YCbCr2RGB(image).astype('uint8')

    def build_decoders(self, M, N, levels, thresholds):
        decoders = []
        for i in range(3):
            max_thresh = thresholds[i]
            if i == 0:
                decoders.append(ZeroTreeDecoder(M, N, levels, max_thresh, WAVELET))
            else:
                decoders.append(ZeroTreeDecoder(M // 2, N // 2, levels - 1, max_thresh, WAVELET))
        return decoders