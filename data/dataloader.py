#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def load_data(filename):
    loaded = np.load(filename)

    xtr = loaded['xtr']
    ytr = loaded['ytr']
    xva = loaded['xva']
    yva = loaded['yva']
    xte = loaded['xte']
    yte = loaded['yte']

    return (xtr, ytr, xva, yva, xte, yte)
