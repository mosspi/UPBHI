#!/usr/bin/env python
"""
Author: Martijn Prevoo
Date: 20-03-2025

Function: Create k-mer composition features from a folder counting k-mer counts

Usage: python kmer_features.py <path/to/input_folder> <path/to/output_file.csv>

    input_folder: Location of a folder containing files with k-mer counts.
                  Each file should contain the counts for an individual strain.

    output_file.csv: Location where to write the outputted k-mer features.

"""
import os
import pandas as pd
from sys import argv


def parse_kmer(file, a_dict: dict = None):
    """"""
    if not a_dict:
        a_dict = {}
    with open(file, 'r') as f:
        name = file.rsplit('/', 1)[-1].rsplit('.', 1)[0]
        line = f.readline()
        while line:
            kmer, count = line.strip().split('\t')
            if kmer not in a_dict:
                a_dict[kmer] = {}
            a_dict[kmer][name] = count
            line = f.readline()
    return a_dict


def kmer_features(folder):
    """"""
    kmers = {}
    n = 0
    for file in os.listdir(folder):
        kmers = parse_kmer(f'{folder}/{file}', kmers)
        n += 1
    kmers = pd.DataFrame(kmers).fillna(0).T.astype(int)
    print(folder, kmers.values.max())
    kmers = kmers.div(kmers.sum(axis=1), axis=0) * 100
    return kmers


def xy(i, pf, bf):
    """"""
    x = {}
    y = i['interaction']
    for inter, (_, phage, bacterium, _, _) in i.iterrows():
        x[inter] = dict(abs(pf[phage] - bf[bacterium]))
    x = pd.DataFrame(x).T
    return x, y


def main():
    """"""
    input_folder, output_folder = argv[1:]
    kmer_features(input_folder).to_csv(output_folder)


if __name__ == '__main__':
    main()
