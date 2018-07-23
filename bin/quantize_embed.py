from __future__ import absolute_import, division, print_function

import sys
sys.path.append(".")
import os
import time
import numpy as np
from argparse import ArgumentParser
from nncompress import EmbeddingCompressor

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("--matrix", default="data/word2vec_100_3.npy")
    ap.add_argument("--model", default="data/mymodel")
    ap.add_argument("--limit", default=200000, type=int)
    ap.add_argument("-M", "--M", default=64, type=int)
    ap.add_argument("-K", "--K", default=32, type=int)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    args = ap.parse_args()

    matrix = np.load(args.matrix)
    if args.limit > 0:
        matrix = matrix[:args.limit]

    compressor = EmbeddingCompressor(args.M, args.K, args.model)
    if args.train:
        compressor.train(matrix)
    elif args.export:
        compressor.export(matrix, args.model, args.matrix)
    elif args.evaluate:
        distance = compressor.evaluate(matrix)
        print("Mean euclidean distance:", distance)
