#!/usr/bin/env python3
"""
Author: Martijn Prevoo

Function: ###
Usage: ###

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
    arg = argv[1:]
    kmer_folder = arg[0]
    k = kmer_folder.rsplit('mers', 1)[0].rsplit('/', 1)[-1]

    phage_kmers = f'{kmer_folder}phage_{k}mers/'
    bact_kmers = f'{kmer_folder}bacterium_{k}mers/'
    feature_folder = f'input/model_data/{k}mer/'

    if not os.path.exists(feature_folder):
        os.mkdir(feature_folder)
    pf_file = f'{feature_folder}{k}mer_phages_features.csv'
    bf_file = f'{feature_folder}{k}mer_bacteria_features.csv'
    kmer_features(phage_kmers).to_csv(pf_file)
    kmer_features(bact_kmers).to_csv(bf_file)


if __name__ == '__main__':
    main()
