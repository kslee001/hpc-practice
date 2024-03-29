import argparse
import itertools, functools
import array
import os
import struct
import numpy as np

def run(args):
    with open(args.bin1, 'rb') as f1, open(args.bin2, 'rb') as f2:
        bin1, bin2 = f1.read(), f2.read()

    if len(bin1) != len(bin2):
        print(f'Sizes are different (bin1={len(bin1)}, bin2={len(bin2)})')
        return

    bin1 = np.asarray(struct.unpack(f'{len(bin1) // 4}f', bin1))
    bin2 = np.asarray(struct.unpack(f'{len(bin2) // 4}f', bin2))

    diff = bin1 - bin2
    print(f'Max of difference: {diff.max()}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bin1', help='Binary to compare. (e.g., output1.bin)')
    parser.add_argument('bin2', help='Binary to compare. (e.g., output2.bin)')
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()
