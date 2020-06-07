import numpy as np
import pywt
from queue import Queue
from bitarray import bitarray

PREFIX_FREE_CODE = {
    "T": bitarray('0'),
    "Z": bitarray('10'),
    "P": bitarray('110'),
    "N": bitarray('111')
}
class CoefficientTree:
    def __init__(self, value, level, quadrant, loc, children=[]):
        self.value = value
        self.level = level
        self.quadrant = quadrant
        self.children = children
        self.loc = loc
        self.code = None
        
    def zero_code(self, threshold):
        for child in self.children:
            child.zero_code(threshold)
            
        if abs(self.value) >= threshold:
            self.code = "P" if self.value > 0 else "N"
        else:
            self.code = "Z" if any([child.code != "T" for child in self.children]) else "T"


    @classmethod
    def build_trees(cls, coeffs):
        def build_children(level, loc, quadrant):
            if level + 1 > len(coeffs): return []

            i, j = loc
            child_locs = [(2*i, 2*j), (2*i, 2*j + 1), (2*i + 1, 2*j), (2*i + 1, 2*j + 1)]
            children = []
            for cloc in child_locs:
                if cloc[0] >= coeffs[level][quadrant].shape[0] or cloc[1] >= coeffs[level][quadrant].shape[1]:
                    continue
                node = CoefficientTree(coeffs[level][quadrant][cloc], level, quadrant, cloc)
                node.children = build_children(level + 1, cloc, quadrant)
                children.append(node)
            return children

        LL = coeffs[0]

        LL_trees = []
        for i in range(LL.shape[0]):
            for j in range(LL.shape[1]):
                children = [CoefficientTree(subband[i, j], 1, quad, (i,j), children=build_children(2, (i,j), quad))
                            for quad, subband in enumerate(coeffs[1])]
                
                LL_trees.append(CoefficientTree(LL[i,j], 0, None, (i,j), children=children))

        return LL_trees

class ZeroTreeScan():
    def __init__(self, code, isDominant):
        self.isDominant = isDominant
        self.code = code
        self.bits = code if not isDominant else self.rle_bits(code)

    def __len__(self):
        return len(self.bits)

    def tofile(self, file, padto=16):
        bits = self.bits.copy()

        if padto != 0 and len(bits) % padto != 0:
            bits.extend([False for _ in range(padto - (len(bits) % padto))])
            
        bits.tofile(file)

    def rle_bits(self, code):
        bitarr = bitarray()
        bitarr.encode(PREFIX_FREE_CODE, code)
        return bitarr

    @classmethod
    def from_bits(cls, bits, isDominant):
        code = bits.decode(PREFIX_FREE_CODE) if isDominant else bits
        return ZeroTreeScan(code, isDominant)

class ZeroTreeEncoder:
    def __init__(self, image, wavelet):
        coeffs = pywt.wavedec2(image, wavelet)
        coeffs = self.quantize(coeffs)

        self.trees = CoefficientTree.build_trees(coeffs)
        
        coeff_arr, _ = pywt.coeffs_to_array(coeffs)
        
        self.thresh = np.power(2, np.floor(np.log2(np.max(np.abs(coeff_arr)))))
        self.start_thresh = self.thresh

        self.secondary_list = None
        self.perform_dominant_pass = True

    def __iter__(self):
        return self

    def __next__(self):
        if self.thresh <= 0: raise StopIteration
        if self.thresh <= 1 and not self.perform_dominant_pass: raise StopIteration
        
        if self.perform_dominant_pass:
            scan, next_coeffs = self.dominant_pass()

            if self.secondary_list is None:
                self.secondary_list = next_coeffs
            else:
                self.secondary_list = np.concatenate((self.secondary_list, next_coeffs))
                
            self.perform_dominant_pass = False
            return scan
        else:
            scan = self.secondary_pass()
            self.thresh //= 2
            self.perform_dominant_pass = True
            return scan

    def quantize(self, subbands):
        quant = lambda q: np.sign(q) * np.floor(np.abs(q))
            
        quantized = []
        for i, subband in enumerate(subbands):
            if isinstance(subband, tuple):
                quantized.append(tuple([quant(sb) for sb in subband]))
            else:
                quantized.append(quant(subband))
        return quantized
       
    def dominant_pass(self):
        sec = []
        
        q = Queue()
        for parent in self.trees:
            parent.zero_code(self.thresh)
            q.put(parent)
            
        codes = []
        while not q.empty():
            node = q.get()
            codes.append(node.code)
            
            if node.code != "T":
                for child in node.children:
                    q.put(child)
                    
            if node.code == "P" or node.code == "N":
                sec.append(node.value)
                node.value = 0

        return ZeroTreeScan(codes, True), np.abs(np.array(sec))
    
    def secondary_pass(self):
        bits = bitarray()
        
        middle = self.thresh // 2
        for i, coeff in enumerate(self.secondary_list):
            if coeff - self.thresh >= 0:
                self.secondary_list[i] -= self.thresh
            bits.append(self.secondary_list[i] >= middle)

        return ZeroTreeScan(bits, False)

class ZeroTreeDecoder:
    def __init__(self, M, N, start_thres, wavelet):
        img = np.zeros((M, N))
        self.wavelet = wavelet
        self.coeffs = pywt.wavedec2(img, wavelet)
        self.trees = CoefficientTree.build_trees(self.coeffs)
        self.T = start_thres
        self.processed = []

    def getImage(self):
        return pywt.waverec2(self.coeffs, self.wavelet)

    def process(self, scan):
        if scan.isDominant:
            self.dominant_pass(scan.code)
        else:
            self.secondary_pass(scan.code)
                        
    def dominant_pass(self, code_list):
        q = []
        for parent in self.trees:
            q.append(parent)
            
        for code in code_list:
            if len(q) == 0:
                break
            node = q.pop(0)
            if code != "T":
                for child in node.children:
                    q.append(child)
            if code == "P" or code == "N":
                node.value = (1 if code == "P" else -1) * self.T
                self._fill_coeff(node)
                self.processed.append(node)
                
    def secondary_pass(self, bitarr):
        if len(bitarr) != len(self.processed):
            bitarr = bitarr[:len(self.processed)]
        for bit, node in zip(bitarr, self.processed):
            if bit:
                node.value += (1 if node.value > 0 else -1) * self.T // 2
                self._fill_coeff(node)
                
        self.T //= 2
        
    def _fill_coeff(self, node):
        if node.quadrant is not None:
            self.coeffs[node.level][node.quadrant][node.loc] = node.value
        else:
            self.coeffs[node.level][node.loc] = node.value

    def urle(self, rle_code):
        return rle_code.decode(PREFIX_FREE_CODE)