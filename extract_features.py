#!/usr/bin/env python3
"""
Author: Martijn Prevoo

Function: ###
Usage: ###

"""
import os
from sys import argv
import pandas as pd


class Features:
    def __init__(self, name: str = 'features'):
        """"""
        self.name = name
        self.genomes = None
        self.groups = None
        self.features = None
        self.feature_info = None
        self.overview = None
        self.proteins = None
        self.group_info = None

    def __str__(self):
        """"""
        if not self.overview:
            ov = 'total: 0'
        else:
            ov = ' | '.join([f'{k}: {v}' for k, v in self.overview.items()])
        return f'{self.name} - {ov}'

    def from_db(self, db_path: str, copy_number: bool = True):
        """"""
        genomes_path = f'{db_path}/databases/genome.db/genomes.txt'
        groups_path = f'{db_path}/pantools_homology_groups.txt'
        proteins_path = f'{db_path}/proteins/'
        classified_groups_path = (f'{db_path}/gene_classification/'
                                  f'classified_groups.csv')
        if not os.path.exists(classified_groups_path):
            classified_groups_path = None
        self.from_files(genomes_path, groups_path, classified_groups_path,
                        proteins_path, copy_number)
        return self

    def from_files(self, genomes_path: str = None, groups_path: str = None,
                   classified_groups_path: str = None,
                   proteins_path: str = None, copy_number: bool = True):
        """"""
        if genomes_path:
            self.genomes = self.parse_genomes(genomes_path)
        if groups_path:
            self.groups = self.parse_groups(groups_path)
        if proteins_path:
            self.proteins = self.parse_proteins(proteins_path)
        self.group_info = self.get_group_info()
        self.features, self.feature_info, self.overview = (
            self.cg_to_features(classified_groups_path, copy_number))
        return self

    @staticmethod
    def parse_genomes(file: str):
        """"""
        genomes = {}
        with open(file, 'r') as read:
            for genome in read.read().split('Genome')[1:]:
                number, file = genome.split('\n')[:2]
                number = number.strip()
                file = file.split(' ', 2)[-1].split('/')[-1].rsplit('.', 1)[0]
                genomes[number] = file
        return genomes

    @staticmethod
    def parse_groups(file: str):
        """"""
        groups = {}
        with open(file, 'r') as read:
            line = read.readline()
            while line:
                if not line.startswith('#'):
                    group, proteins = line.strip().split(': ')
                    proteins = [p.rsplit('#', 1)[0]
                                for p in proteins.split(' ')]
                    groups[group] = proteins
                line = read.readline()
        return groups

    def parse_proteins(self, folder: str):
        """"""
        proteins = {}
        for file in os.listdir(folder):
            file = f'{folder}/{file}'
            if file.endswith('.fasta'):  # protein files
                proteins = {**proteins, **self.parse_protein_file(file)}
            elif file.endswith('.gff') or file.endswith('.gff3'):  # gff file
                proteins = {**proteins, **self.parse_annotation(file)}
        return proteins

    def parse_protein_file(self, file: str):
        """"""
        data = {}
        strain = file.rsplit('_', 1)[-1].rsplit('.', 1)[0]
        if self.genomes:
            strain = self.genomes[strain]
        with open(file, 'r') as read:
            line = read.readline()
            while line:
                if line.startswith('>'):
                    line = line.strip()[1:]
                    data[line] = strain
                line = read.readline()
        return data

    @staticmethod
    def parse_annotation(file: str):
        """"""
        data = {}
        strain = file.rsplit('/')[-1].rsplit('.', 1)[0]
        with open(file, 'r') as read:
            line = read.readline()
            while line:
                if not line.startswith('#'):
                    ident = line.strip().rsplit(
                        '\t', 1)[-1].split('ID=', 1)[-1].split(';', 1)[0]
                    data[ident] = strain
                line = read.readline()
        return data

    def get_group_info(self):
        """"""
        if not self.groups or not self.proteins:
            return None
        info = {}
        for group, ps in self.groups.items():
            ps = [f'{self.proteins[p]}={p}' for p in ps]
            info[group] = ps
        return info

    def groups_to_cg(self):
        """"""
        if not self.groups or not self.proteins:
            return None
        strains = list(set(self.proteins.values()))
        cg = pd.DataFrame(0, index=self.groups.keys(), columns=strains)
        cg.insert(0, 'class', '')
        for group, ps in self.groups.items():
            ps = [self.proteins[p] for p in ps]
            for p in ps:
                cg.loc[group, p] += 1
            n = len(set(ps))
            if n == len(strains):
                group_class = 'core'
            elif n > 1:
                group_class = 'accessory'
            else:
                group_class = 'unique'
            cg.loc[group, 'class'] = group_class
        return cg.sort_index()

    def parse_classified_groups(self, file: str = None):
        """"""
        if file:
            cg = pd.read_csv(file, index_col=0, header=0).fillna(0)
        elif self.groups and self.proteins:
            cg = self.groups_to_cg()
        else:
            return None, None
        # create overview
        overview = {'total': len(cg),
                    'core': len(cg[cg['class'].str.contains('core')]),
                    'unique': len(cg[cg['class'].str.contains('unique')])}
        # remove non-accessory groups
        cg = cg[cg['class'].str.contains('accessory')]
        overview['accessory'] = len(cg)
        # gename columns
        cg = cg.drop(columns='class')
        if file and self.genomes:
            cg.columns = [self.genomes[c.rsplit(' ', 1)[-1]]
                          for c in cg.columns]
        return cg, overview

    @staticmethod
    def get_features(cg):
        """"""
        fingerprints = {}
        for hg, features in cg.iterrows():
            fp = ','.join(list(map(str, features.tolist())))
            if fp not in fingerprints:
                fingerprints[fp] = [str(hg)]
            else:
                fingerprints[fp].append(str(hg))
        # features
        features = []
        info = {}
        for i, (fp, hgs) in enumerate(fingerprints.items()):
            features.append(list(map(int, fp.split(','))))
            info[f'f{int(i) + 1}'] = [len(hgs)] + hgs
        features = pd.DataFrame(features, columns=cg.columns,
                                index=list(info.keys()))
        return features, info

    def cg_to_features(self, file: str = None, copy_number: bool = True):
        """"""
        cg, overview = self.parse_classified_groups(file)
        if cg is None or not overview:
            return None, None, None
        if not copy_number:
            cg = cg.mask(cg > 0, 1)
        features, feature_info = self.get_features(cg)
        overview['features'] = len(features)
        return features, feature_info, overview

    def to_folder(self, folder: str = 'features'):
        """"""
        # create folder if it does not exist
        if not os.path.exists(folder):
            os.mkdir(folder)
        # write data to folder
        if self.features is not None:
            self.features.to_csv(f'{folder}/{self.name}_features.csv')
        if self.feature_info is not None:
            with open(f'{folder}/{self.name}_feature_info.txt', 'w') as out:
                for k, v in self.feature_info.items():
                    out.write(f'{k},{v[0]},{";".join(v[1:])}\n')
        if self.overview is not None:
            with open(f'{folder}/{self.name}_overview.txt', 'w') as out:
                out.write(self.__str__())
        if self.group_info is not None:
            with open(f'{folder}/{self.name}_group_info.txt', 'w') as out:
                for k, v in self.group_info.items():
                    out.write(f'{k},{";".join(v)}\n')


class Interactions:
    def __init__(self):
        """"""
        self.phage_genomes = None
        self.bacteria_genomes = None
        self.phage_taxa = None
        self.bacteria_taxa = None
        self.interactions = None

    def __str__(self):
        return str(self.interactions)

    def from_files(self, matrix_path: str,
                   phage_taxa_path: str = None,
                   bacteria_taxa_path: str = None,
                   phage_genomes_path: str = None,
                   bacteria_genomes_path: str = None):
        """"""
        if phage_genomes_path:
            phage_genomes = Features.parse_genomes(phage_genomes_path)
            self.phage_genomes = list(phage_genomes.values())
        if bacteria_genomes_path:
            bacteria_genomes = Features.parse_genomes(bacteria_genomes_path)
            self.bacteria_genomes = list(bacteria_genomes.values())
        if phage_taxa_path:
            self.phage_taxa = self.parse_taxa(phage_taxa_path)
        if bacteria_taxa_path:
            self.bacteria_taxa = self.parse_taxa(bacteria_taxa_path)
        matrix = self.parse_matrix(matrix_path)
        self.interactions = self.create_interactions(matrix)
        return self

    @staticmethod
    def parse_taxa(file):
        """"""
        data = {}
        with open(file, 'r') as read:
            line = read.readline()
            while line:
                line = line.strip().split(',')
                data[line[0]] = line[1]
                line = read.readline()
        return data

    def parse_matrix(self, file):
        """"""
        matrix = pd.read_csv(file, header=0, index_col=0)
        if self.phage_genomes or self.bacteria_genomes:
            # remove rows not present in phage and bacteria genomes
            for row in matrix.index:
                if (row not in self.phage_genomes and
                        row not in self.bacteria_genomes):
                    matrix = matrix.drop(index=row)
            # remove columns not present in phage and bacteria genomes
            for column in matrix.columns:
                if (column not in self.phage_genomes and
                        column not in self.bacteria_genomes):
                    matrix = matrix.drop(columns=column)
            # make sure rows are bacteria and columns phages
            if (matrix.index[0] in self.phage_genomes or
                    matrix.columns[0] in self.bacteria_genomes):
                matrix = matrix.T
        elif self.phage_taxa or self.bacteria_taxa:
            if (matrix.index[0] in self.phage_taxa or
                    matrix.columns[0] in self.bacteria_taxa):
                matrix = matrix.T
        return matrix

    def create_interactions(self, matrix):
        """"""
        index = []
        inter = {'interaction': [], 'phage': [], 'bacterium': []}
        if self.phage_taxa:
            inter['phage_taxa'] = []
        if self.bacteria_taxa:
            inter['bacterium_taxa'] = []
        i = 0
        for phage in matrix.columns:
            for bacterium in matrix.index:
                i += 1
                index.append(f'i{i}')
                inter['interaction'].append(matrix.loc[bacterium, phage])
                inter['phage'].append(phage)
                inter['bacterium'].append(bacterium)
                if self.phage_taxa:
                    inter['phage_taxa'].append(self.phage_taxa[phage])
                if self.bacteria_taxa:
                    inter['bacterium_taxa'].append(
                        self.bacteria_taxa[bacterium])
        return pd.DataFrame(inter, index=index)


def arg_value(arg: list, possible_arguments: list):
    """"""
    for a in possible_arguments:
        if a in arg:
            return arg[arg.index(a) + 1]
    return None


def create_features(arg: list):
    """"""
    name = arg_value(arg, ['-n', '--name'])
    output = arg_value(arg, ['-o', '--output'])
    if arg_value(arg, ['-cn']) in ('n', 'no'):
        cn = False
    else:
        cn = True
    # create features
    if '-db' in arg:  # from db
        db = arg[arg.index('-db') + 1]
        features = Features(name).from_db(db_path=db, copy_number=cn)
    else:  # from files
        ge = arg_value(arg, ['-ge', '--genomes'])
        gr = arg_value(arg, ['-gr', '--groups'])
        cg = arg_value(arg, ['-cg', '--classified_groups'])
        p = arg_value(arg, ['-p', '-proteins'])
        features = Features(name).from_files(genomes_path=ge, groups_path=gr,
                                             classified_groups_path=cg,
                                             proteins_path=p, copy_number=cn)
    print(features)
    features.to_folder(output)


def create_interactions(arg):
    """"""
    m = arg_value(arg, ['-m', '--matrix'])
    output = arg_value(arg, ['-o', '--output'])
    pt = arg_value(arg, ['-pt', '--phage_taxa'])
    bt = arg_value(arg, ['-bt', '--bacteria_taxa'])
    pg = arg_value(arg, ['-pg', '--phage_genomes'])
    bg = arg_value(arg, ['-bg', '--bacteria_genomes'])
    inter = Interactions().from_files(
        matrix_path=m,
        phage_taxa_path=pt,
        bacteria_taxa_path=bt,
        phage_genomes_path=pg,
        bacteria_genomes_path=bg)
    inter.interactions.to_csv(output)


def run_scripts(arg: list):
    """"""
    # example1: -f -db (--database) <db_folder>
    #              -cn (--copy_number) <y(es), n(o): default is y>
    #              -n (--name) <str>
    #              -o (--output) <output_folder>
    # example2: -f -ge (--genomes) <genomes.txt>
    #              -gr (--groups) <groups.txt>
    #              -cg (--classified_groups) <cg.csv>
    #              -p (--proteins) <protein_folder>
    #              -cn (--copy_number) <y(es), n(o): default is y>
    #              -n (--name) <str>
    #              -o (--output) <output_folder>
    # example3: -i -m (--matrix) <matrix.csv>
    #              -pt (--phage_taxa) <phage_taxa.csv>
    #              -bt (--bacteria_taxa) <bacteria_taxa.csv>
    #              -pg (--phage_genomes) <phage_genomes.txt>
    #              -bg (--bacteria_genomes) <bacteria_genomes.txt>
    #              -o (--output) <output_file>
    if '-f' in arg or '--features' in arg:
        create_features(arg)
    elif '-i' in arg or '--interactions' in arg:
        create_interactions(arg)
    else:
        exit('ERROR: must either specify either '
             '--features (-f) or --interactions (-i)')


def do_thing(in_folder: str, out_folder: str,
             relax_modes: list = ('3', '4', '5', '6', '7'),
             prefix: str = '', copy_number: bool = True):
    """"""
    # out_folder = 'output/updated_gpa/'
    # in_folder = 'input/updated_HG_data'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    arg = ['-i', '-m', f'{in_folder}/infection_matrix.csv',
           '-o', f'{out_folder}/interactions.csv',
           '-pt', f'{in_folder}/phenotypes_phages.csv',
           '-bt', f'{in_folder}/phenotypes_bacteria.csv',
           '-pg', f'{in_folder}/{prefix}phages_genomes.txt',
           '-bg', f'{in_folder}/{prefix}bacteria_genomes.txt']
    run_scripts(arg)
    if copy_number:
        cn = 'y'
    else:
        cn = 'n'
    for r in relax_modes:
        o = f'{out_folder}/{r}_{prefix}hg'
        # phages
        s = 'phages'
        arg = ['-f', '-n', f'{r}_{prefix}{s}',
               '-o', o,
               '-cn', cn,
               '-ge', f'{in_folder}/{prefix}{s}_genomes.txt',
               '-gr', f'{in_folder}/{prefix}{s}_{r}_groups.txt',
               '-cg', f'{in_folder}/{prefix}{s}_{r}_classified_groups.csv',
               '-p', f'input/annotations/{s}']
        run_scripts(arg)
        # bacteria
        s = 'bacteria'
        arg = ['-f', '-n', f'{r}_{prefix}{s}',
               '-o', o,
               '-cn', cn,
               '-ge', f'{in_folder}/{prefix}{s}_genomes.txt',
               '-gr', f'{in_folder}/{prefix}{s}_{r}_groups.txt',
               '-cg', f'{in_folder}/{prefix}{s}_{r}_classified_groups.csv',
               '-p', f'input/annotations/{s}']
        run_scripts(arg)


def main():
    """"""
    in_folder = 'input/old_HG_data'
    out_folder = 'output/medium_dataset/'
    do_thing(relax_modes=['R7'],
             in_folder=in_folder, out_folder=out_folder, prefix='medium_')


if __name__ == '__main__':
    main()
