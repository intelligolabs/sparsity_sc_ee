#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from argparse import ArgumentParser

from build_and_run import build_and_run_model


def main(args):
    fo = np.array(args.fo)
    config = np.array(args.config)

    if len(config) - 1 != len(fo):
        print("Invalid config: the len of 'config' should be one element longer than the 'fo'!")
        return

    build_and_run_model(config, fo, args.dataset_path)


if __name__ == '__main__':
    parser = ArgumentParser('Predefined sparsity in SC and EE')

    parser.add_argument('-f', '--fo', nargs='+', type=int, required=True)
    parser.add_argument('-c', '--config', nargs='+', type=int, required=True)
    parser.add_argument('-s', '--dataset_path', type=str, default='datasets/mnist.npz')

    args = parser.parse_args()

    main(args)
