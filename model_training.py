#!/usr/bin/env python
"""
Author: Martijn Prevoo

Function: Train and validate models predicting phage-host interactions

Usage: python model_training.py <threads>

"""
import warnings
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sys import argv
from statistics import mean, stdev
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             RocCurveDisplay, auc, confusion_matrix)
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.ensemble import RandomForestClassifier


class ModelTraining:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        # empty variables
        self.i = None
        self.pf = None
        self.bf = None
        self.datatype = None  # 0=unique, 1=difference, 2=distance
        self.kfolds = None
        self.fitted_models = None

    def __str__(self):
        return f'{self.name}: {self.model}'

    def import_data(self, interaction_file: str, pf_file: str, bf_file: str,
                    datatype: int = 1):
        """

        :param interaction_file:
        :param pf_file:
        :param bf_file:
        :param datatype: 0 (concatenate), 1 (difference), 2 (distance)
        :return:
        """
        self.i = pd.read_csv(interaction_file, header=0, index_col=0)
        self.pf = pd.read_csv(pf_file, header=0, index_col=0)
        self.bf = pd.read_csv(bf_file, header=0, index_col=0)
        self.datatype = datatype
        # if datatype is 1, or 2 (dif or dist) make indexes the same
        if self.datatype > 0 and datatype < 3:
            if set(self.pf.index) != set(self.bf.index):
                pf_add = list(set(self.bf.index) - set(self.pf.index))
                bf_add = list(set(self.pf.index) - set(self.bf.index))
                for i in pf_add:
                    self.pf.loc[i] = [0] * len(self.pf.columns)
                for i in bf_add:
                    self.bf.loc[i] = [0] * len(self.bf.columns)
            # make sure data is in same order
            self.pf = self.pf.loc[self.bf.index, :]

    def check_if_data(self):
        """"""
        missing = []
        # check data
        if self.i is None:
            missing.append('interactions')
        if self.pf is None:
            missing.append('phage_features')
        if self.bf is None:
            missing.append('bacteria_features')
        # return
        if missing:
            exit(f"ERROR: missing data for: {', '.join(missing)}")

    def _xy_(self, specifics: (str, list) = None):
        """"""
        if specifics:
            column, specifics = specifics
            i = self.i[self.i[column].isin(specifics)]
        else:
            i = self.i
        # get classes
        y = i['interaction']
        # get features based on datatype
        if self.datatype == 1:
            x = self._x_difference_(i)  # difference
        elif self.datatype == 2:
            x = self._x_distance_(i)  # distance
        else:
            x = self._x_unique_(i)  # unique
        return x, y

    def _x_unique_(self, i):
        """"""
        x_header = ([f'p{f}' for f in self.pf.index] +
                    [f'b{f}' for f in self.bf.index])
        x = {}
        for inter, data in i.iterrows():
            try:
                phage_data = self.pf[data['phage']].tolist()
                bact_data = self.bf[data['bacterium']].tolist()
                x[inter] = phage_data + bact_data
            except KeyError:  # if phages or bacterium not in features
                pass
        x = pd.DataFrame.from_dict(x).T
        x.columns = x_header
        return x

    def _x_difference_(self, i):
        """"""
        x = {}
        for inter, data in i.iterrows():
            phage, bacterium = data['phage'], data['bacterium']
            x[inter] = dict(abs(self.pf[phage] - self.bf[bacterium]))
        x = pd.DataFrame(x).T
        return x

    def _x_distance_(self, i):
        """"""
        x = {}
        for inter, data in i.iterrows():
            phage = self.pf[data['phage']].to_numpy()
            bacterium = self.bf[data['bacterium']].to_numpy()
            x[inter] = np.linalg.norm(phage - bacterium)  # euclidian distance
        x = pd.DataFrame(x, index=['distance']).T
        return x

    def set_kfolds(self, k: int = 5, shuffle: bool = True,
                   random_state: int = None):
        """"""
        self.kfolds = StratifiedKFold(n_splits=k, shuffle=shuffle,
                                      random_state=random_state)

    def fit(self, shuffle: bool = True, stratify: bool = True,
            test_size: float = .2, random_state: int = None):
        """"""
        x, y = self._xy_()
        if stratify:
            stratify = y
        else:
            stratify = None
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, shuffle=shuffle, stratify=stratify, test_size=test_size,
            random_state=random_state)
        fits = self.model.fit(x_train, y_train)
        self.fitted_models = [(fits, x_test, y_test)]

    def cv_fit(self, k: int = 5, shuffle: bool = True,
               random_state: int = None):
        """"""
        self.fitted_models = []  # reset fitted models
        self.check_if_data()
        x, y = self._xy_()
        if not self.kfolds:
            self.set_kfolds(k=k, shuffle=shuffle, random_state=random_state)
        # fit models
        for i_train, i_test in self.kfolds.split(x, y):
            x_train, x_test = x.iloc[i_train], x.iloc[i_test]
            y_train, y_test = y.iloc[i_train], y.iloc[i_test]
            model = clone(self.model)
            fits = model.fit(x_train, y_train)
            self.fitted_models.append((fits, x_test, y_test))

    def loo_fit(self, column: str, test_set: str):
        """"""
        self.fitted_models = []  # reset fitted models
        self.check_if_data()
        # fit model for each unique value in column
        values = list(set(self.i[column]))
        i = values.index(test_set)
        train_set = values[:i] + values[i + 1:]
        test_set = [values[i]]
        x_train, y_train = self._xy_(specifics=(column, train_set))
        x_test, y_test = self._xy_(specifics=(column, test_set))
        model = clone(self.model)
        fits = model.fit(x_train, y_train)
        self.fitted_models.append((fits, x_test, y_test))

    def scores(self, scores: dict = None):
        """"""
        if not scores:
            scores = {'accuracy': accuracy_score, 'precision': precision_score,
                      'recall': recall_score, 'f1': f1_score}
        scoring = {s: [] for s in scores.keys()}
        for fitted, x_test, y_test in self.fitted_models:
            y_pred = fitted.predict(x_test)
            for name, score in scores.items():
                scoring[name].append(score(y_test, y_pred))

        scores = []
        for k, v in scoring.items():
            s = 0
            if len(self.fitted_models) > 1:
                s = round(stdev(v), 4)
                v = [mean(v)]
            v = round(v[0], 4)
            scores.append(f'{k}={v}({s})')
        return f"{self.name} - {' '.join(scores)}"

    def roc_values(self):
        """"""
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []
        for fitted, x_test, y_test in self.fitted_models:
            ax = plt.gca()  # create temp axis
            viz = RocCurveDisplay.from_estimator(fitted, x_test, y_test, ax=ax)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = .0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
            ax.remove()  # remove temp axis
        return mean_fpr, tprs, aucs

    def __feature_strain_taxa__(self, features, taxa: bool = True):
        """"""
        strains_list = []
        taxa_list = []
        for f in features:
            # check if feature is from phage or bacteria
            if f.startswith('p'):
                species, column, sf = 'phage', 'phage_taxa', self.pf
            elif f.startswith('b'):
                species, column, sf = 'bacterium', 'bacterium_taxa', self.bf
            else:  # if feature is from neither append NaN
                strains_list.append('NaN')
                taxa_list.append('NaN')
                continue
            strains = sf.columns[sf.loc[f[1:]] > 0].tolist()
            if taxa and column in self.i:
                taxa = set(self.i[column][self.i[species].isin(strains)])
            else:
                taxa = ['NA']
            strains_list.append(';'.join(strains))
            taxa_list.append(';'.join(taxa))
        return strains_list, taxa_list

    def feature_importance(self,
                           strain_info: bool = True, taxa_info: bool = False):
        """"""
        features = None
        scores = None
        # get importance scores
        for fitted, x_test, _ in self.fitted_models:
            if not features or not scores:
                features = x_test.columns.tolist()
                scores = {f: [] for f in features}
            for i, f in enumerate(fitted.feature_importances_):
                scores[features[i]].append(f)
        # get mean score and its std dev
        for f in scores:
            s = scores[f]
            sd = 0
            if len(s) > 1:
                sd = stdev(s)
                s = [mean(s)]
            scores[f] = {'importance': s[0], 'stdev': sd}
        fi = pd.DataFrame(scores).T
        if strain_info or taxa_info:
            strains, taxa = self.__feature_strain_taxa__(fi.index,
                                                         taxa=taxa_info)
            if taxa_info:
                fi['taxa'] = taxa
            if strain_info:
                fi['strains'] = strains
        return fi.sort_values(by='importance', ascending=False)

    def misclassified(self, interactions: bool = True,
                      taxa: bool = False, strains: bool = False):
        """"""
        # create dict
        misclassified = {}
        ex = {'tp': 0, 'fn': 0, 'fp': 0, 'tn': 0}
        if taxa:
            pt = {t: ex.copy()
                  for t in sorted(list(set(self.i['phage_taxa'])))}
            bt = {t: ex.copy()
                  for t in sorted(list(set(self.i['bacterium_taxa'])))}
            misclassified = {**misclassified, **pt, **bt}
        if strains:
            ps = {s: ex.copy() for s in self.pf.columns}
            bs = {s: ex.copy() for s in self.bf.columns}
            misclassified = {**misclassified, **ps, **bs}
        if interactions:
            misclassified = {**misclassified,
                             **{i: ex.copy() for i in self.i.index}}
        # parse fitted models
        for fitted, x_test, y_test in self.fitted_models:
            inters = x_test.index.tolist()
            y_test = y_test.tolist()
            y_pred = fitted.predict(x_test).tolist()
            # count predictions
            for i, pred in enumerate(y_pred):
                inter_data = self.i.loc[inters[i]]
                # determine if prediction is false or true positive / negative
                key = 't'
                if pred != y_test[i]:
                    key = 'f'
                if pred == 1:
                    key = f'{key}p'
                else:
                    key = f'{key}n'
                # count prediction
                if interactions:
                    misclassified[inters[i]][key] += 1
                if taxa:
                    misclassified[inter_data['phage_taxa']][key] += 1
                    misclassified[inter_data['bacterium_taxa']][key] += 1
                if strains:
                    misclassified[inter_data['phage']][key] += 1
                    misclassified[inter_data['bacterium']][key] += 1
        return pd.DataFrame(misclassified).T

    @staticmethod
    def __shapely_scores__(shap_values, x, y, fits, feature_list):
        """"""
        sv_df = pd.DataFrame(shap_values, index=x.index, columns=x.columns)
        # order so specific features are first
        if feature_list:
            features = (feature_list +
                        list(sv_df.columns.difference(feature_list)))
            sv_df = sv_df.loc[:, features]
        mean_sv = abs(sv_df).apply(['mean']).T
        mean_sv = mean_sv.sort_values(by='mean', ascending=False)
        return sv_df.loc[:, mean_sv.index].T

    @staticmethod
    def __shapely_summary_plot__(shap_values, x, feature_list, n_display,
                                 title, output):
        """"""
        # only select specific features
        sort = True
        if feature_list:
            shap_values = pd.DataFrame(shap_values, columns=x.columns.tolist())
            shap_values = np.array(shap_values[feature_list])
            x = x[feature_list]
            sort = False
            n_display = len(x)

        # summary plot
        show = True
        if output:
            show = False
        shap.summary_plot(shap_values, x,
                          sort=sort, max_display=n_display, show=show)
        if output:
            plt.savefig(f'{output}.png')
        plt.close()

    def shapely_values(self, feature_list: list = None,
                       n_display: int = 25, output_folder: str = None,
                       include_neg: bool = False):
        """"""
        # fit model
        x, y = self._xy_()
        model = clone(self.model)
        fits = model.fit(x, y)
        # create explainer and get shap values for positive interactions
        explainer = shap.TreeExplainer(model=fits, data=x,
                                       feature_names=x.columns.tolist())
        if include_neg is False:
            x, y = x[y == 1], y[y == 1]  # only positive interactions
        shap_vals = explainer.shap_values(x, y, check_additivity=False)[..., 1]

        # file_name
        if output_folder:
            file_name = f'{output_folder}/{self.name}_shapely'
        else:
            file_name = None

        # summary plot (can only write to output)
        if file_name:
            self.__shapely_summary_plot__(shap_vals, x, feature_list,
                                          n_display,
                                          f'{self.name}', file_name)
        # scores
        sv_df = self.__shapely_scores__(shap_vals, x, y, fits, feature_list)
        if output_folder:
            sv_df.to_csv(f'{file_name}.csv')
        else:
            return sv_df
        return None


def data_files(folder: str, interactions: str = '', features: str = 'R7',
               datatype: int = 0):
    interaction_file = f'{folder}/{interactions}interactions.csv'
    f_folder = f'{folder}/{features}/{features}'
    pf_file = f'{f_folder}_phages_features.csv'
    bf_file = f'{f_folder}_bacteria_features.csv'
    return {'interaction_file': interaction_file,
            'pf_file': pf_file,
            'bf_file': bf_file,
            'datatype': datatype}


def roc_plot_from_values(label, fprs, tprs, aucs, axis=None):
    """"""
    if not axis:
        plt.figure(figsize=(8, 8))
        axis = plt.gca()
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(fprs, mean_tpr)
    std_auc = np.std(aucs)
    axis.plot(fprs, mean_tpr,
              label=f'{label} '
                    f'(AUC = {round(mean_auc, 4)} ({round(std_auc, 4)}))')
    if not axis:
        axis.set(xlabel="False Positive Rate", ylabel="True Positive Rate")
        axis.legend(loc="lower right")
        plt.show()


def cross_validate_roc(folder: str, models: list,
                       k: int = 5, random_state: int = 1,
                       output: str = None,
                       feature_importance: bool = False,
                       misclassified: bool = False,
                       graph_name: str = ''):
    """
    :param folder:
    :param models: list(tuples(str, str, str, int, class))
                 ; folder, interactions, features, datatype, model
    :param k:
    :param random_state:
    :param output:
    :param feature_importance:
    :param misclassified:
    :param graph_name:
    :return:
    """
    fpr_dict = {}
    tpr_dict = {}
    auc_dict = {}
    for name, interactions, features, datatype, model in models:
        model = ModelTraining(name, model)
        model.import_data(**data_files(folder, interactions,
                                       features, datatype))
        model.cv_fit(k=k, random_state=random_state)
        model.scores()
        if output and feature_importance:
            fi = model.feature_importance(strain_info=True, taxa_info=True)
            fi.to_csv(f'{output}/{name}_feature_importance.csv')
        if output and misclassified:
            ms = model.misclassified(interactions=True,
                                     taxa=True, strains=True)
            ms.to_csv(f'{output}/{name}_misclassified.csv')
        fpr, tpr, au = model.roc_values()
        fpr_dict[name] = fpr
        tpr_dict[name] = tpr
        auc_dict[name] = au
    # create plot
    plt.figure(figsize=(8, 8))
    axis = plt.gca()
    for name, _, _, _, _ in models:
        roc_plot_from_values(label=name, axis=axis, fprs=fpr_dict[name],
                             tprs=tpr_dict[name], aucs=auc_dict[name])
    axis.set(xlabel="False Positive Rate", ylabel="True Positive Rate")
    axis.legend(loc="lower right")
    plt.title(graph_name)
    if output:
        graph_name = graph_name.replace(' ', '_')
        plt.savefig(f'{output}/{graph_name}roc.png')
    else:
        plt.show()


def train_rest_test_one(folder: str, interactions: str,
                        features: str, datatype: int = 1,
                        column: str = 'phage_taxa', model=None):
    """"""
    # create model object
    mt = ModelTraining(features, model)
    mt.import_data(
        interaction_file=f'{folder}/{interactions}interactions.csv',
        pf_file=f'{folder}/{features}/{features}_phages_features.csv',
        bf_file=f'{folder}/{features}/{features}_bacteria_features.csv',
        datatype=datatype)

    # train and combined per value in column
    column_values = list(set(mt.i[column]))
    column_values.sort()

    column_values = ['Vibrio lentus', 'Vibrio splendidus']

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        for v in column_values:
            mt.name = v
            mt.loo_fit(column=column, test_set=v)
            # mt.scores()
            mc = mt.misclassified(interactions=False, taxa=True, strains=False)
            # print(v, mc.loc[v].tolist())

            print(datatype, v, '\n', mc, '\n')


def class_weight_models(name, interactions, features, datatype,
                        threads: int = 1, seed: int = None):
    """"""
    # for name, interactions, features, datatype, model in models:
    models = []
    for i in (.01, .1, .2, .3, .4, .5, .6, .7, .8, .9, .99):
        j = round(1 - i, 2)
        data = [f'{name}({i}:{j})', interactions, features, datatype,
                RandomForestClassifier(n_jobs=threads, random_state=seed,
                                       class_weight={1: i, 0: j})]
        models.append(data)
    return models


def weight_distributions(folder, interactions, threads, seed):
    for name, features, datatype in [('HGs', 'R7', 0),
                                     ('com', 'combined', 0),
                                     ('ind', '6mer', 0),
                                     ('dif', '6mer', 1),
                                     ('dis', '6mer', 2)]:
        print(f' -- {name} --')
        mt = ModelTraining('', None)
        mt.import_data(**data_files(folder, interactions, features, datatype))
        for i in (.01, .10, .20, .30, .40, .50, .60, .70, .80, .90, .99):
            j = round(1 - i, 2)
            mt.name = f'{name}({i}:{j})'
            mt.model = RandomForestClassifier(n_jobs=threads,
                                              random_state=seed,
                                              class_weight={1: i, 0: j})
            mt.cv_fit(k=10, shuffle=True, random_state=seed)
            mt.scores()


def main():
    """"""
    # set threads
    try:
        threads = int(argv[1])
    except IndexError:
        threads = 1

    # create model
    seed = 3
    model = RandomForestClassifier(n_jobs=threads, random_state=seed)
    mt = ModelTraining('test', model)

    # import data
    mt.import_data(interaction_file='model_data/partial_interactions.csv',
                   pf_file='model_data/3mer/3mer_phages_features.csv',
                   bf_file='model_data/3mer/3mer_bacteria_features.csv',
                   # datatype: 0 (concatenated), 1 (difference), 2 (distance)
                   datatype=0)

    # train model (either normal fit, k-fold cross validation or leave one out)
    mt.fit(shuffle=True, stratify=True, test_size=.2, random_state=seed)  # fit
    # mt.cv_fit(k=10, shuffle=True, random_state=seed)  # 10-FCV
    # mt.loo_fit(column='phage_taxa', test_set='auto')  # leave one out

    # print - accuracy, precision, recall and F1 score
    print(mt.scores())

    # get true positive / negative and false positive / negative per
    # interaction, taxa and/or strain
    mc = mt.misclassified(interactions=False, taxa=True, strains=False)
    print(mc)  # print misclassified (can also be writen to file 'mc.to_csv()')

    # feature importance with shap (can be writen to file by giving out_folder)
    sv = mt.shapely_values()  # output_folder="path/to/file"
    print(sv)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Author: Martijn Prevoo

Function: ###
Usage: ###

"""
import warnings
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sys import argv
from statistics import mean, stdev
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             RocCurveDisplay, auc, confusion_matrix)
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.ensemble import RandomForestClassifier


class ModelTraining:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        # empty variables
        self.i = None
        self.pf = None
        self.bf = None
        self.datatype = None  # 0=unique, 1=difference, 2=distance
        self.kfolds = None
        self.fitted_models = None

    def __str__(self):
        return f'{self.name}: {self.model}'

    def import_data(self, interaction_file: str, pf_file: str, bf_file: str,
                    datatype: int = 1):
        """

        :param interaction_file:
        :param pf_file:
        :param bf_file:
        :param datatype: 0 for 6mer and 1 for unique
        :return:
        """
        self.i = pd.read_csv(interaction_file, header=0, index_col=0)
        self.pf = pd.read_csv(pf_file, header=0, index_col=0)
        self.bf = pd.read_csv(bf_file, header=0, index_col=0)
        self.datatype = datatype
        # if datatype is 1, or 2 (dif or dist) make indexes the same
        if self.datatype > 0 and datatype < 3:
            if set(self.pf.index) != set(self.bf.index):
                pf_add = list(set(self.bf.index) - set(self.pf.index))
                bf_add = list(set(self.pf.index) - set(self.bf.index))
                for i in pf_add:
                    self.pf.loc[i] = [0] * len(self.pf.columns)
                for i in bf_add:
                    self.bf.loc[i] = [0] * len(self.bf.columns)
            # make sure data is in same order
            self.pf = self.pf.loc[self.bf.index, :]

    def check_if_data(self):
        """"""
        missing = []
        # check data
        if self.i is None:
            missing.append('interactions')
        if self.pf is None:
            missing.append('phage_features')
        if self.bf is None:
            missing.append('bacteria_features')
        # return
        if missing:
            exit(f"ERROR: missing data for: {', '.join(missing)}")

    def _xy_(self, specifics: (str, list) = None):
        """"""
        if specifics:
            column, specifics = specifics
            i = self.i[self.i[column].isin(specifics)]
        else:
            i = self.i
        # get classes
        y = i['interaction']
        # get features based on datatype
        if self.datatype == 1:
            x = self._x_difference_(i)  # difference
        elif self.datatype == 2:
            x = self._x_distance_(i)  # distance
        else:
            x = self._x_unique_(i)  # unique
        return x, y

    def _x_unique_(self, i):
        """"""
        x_header = ([f'p{f}' for f in self.pf.index] +
                    [f'b{f}' for f in self.bf.index])
        x = {}
        for inter, data in i.iterrows():
            try:
                phage_data = self.pf[data['phage']].tolist()
                bact_data = self.bf[data['bacterium']].tolist()
                x[inter] = phage_data + bact_data
            except KeyError:  # if phages or bacterium not in features
                pass
        x = pd.DataFrame.from_dict(x).T
        x.columns = x_header
        return x

    def _x_difference_(self, i):
        """"""
        x = {}
        for inter, data in i.iterrows():
            phage, bacterium = data['phage'], data['bacterium']
            x[inter] = dict(abs(self.pf[phage] - self.bf[bacterium]))
        x = pd.DataFrame(x).T
        return x

    def _x_distance_(self, i):
        """"""
        x = {}
        for inter, data in i.iterrows():
            phage = self.pf[data['phage']].to_numpy()
            bacterium = self.bf[data['bacterium']].to_numpy()
            x[inter] = np.linalg.norm(phage - bacterium)  # euclidian distance
        x = pd.DataFrame(x, index=['distance']).T
        return x

    def set_kfolds(self, k: int = 5, shuffle: bool = True,
                   random_state: int = None):
        """"""
        self.kfolds = StratifiedKFold(n_splits=k, shuffle=shuffle,
                                      random_state=random_state)

    def fit(self, shuffle: bool = True, stratify: bool = True,
            test_size: float = .2, random_state: int = None):
        """"""
        x, y = self._xy_()
        if stratify:
            stratify = y
        else:
            stratify = None
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, shuffle=shuffle, stratify=stratify, test_size=test_size,
            random_state=random_state)
        fits = self.model.fit(x_train, y_train)
        self.fitted_models = [(fits, x_test, y_test)]

    def cv_fit(self, k: int = 5, shuffle: bool = True,
               random_state: int = None):
        """"""
        self.fitted_models = []  # reset fitted models
        self.check_if_data()
        x, y = self._xy_()
        if not self.kfolds:
            self.set_kfolds(k=k, shuffle=shuffle, random_state=random_state)
        # fit models
        for i_train, i_test in self.kfolds.split(x, y):
            x_train, x_test = x.iloc[i_train], x.iloc[i_test]
            y_train, y_test = y.iloc[i_train], y.iloc[i_test]
            model = clone(self.model)
            fits = model.fit(x_train, y_train)
            self.fitted_models.append((fits, x_test, y_test))

    def loo_fit(self, column: str, test_set: str):
        """"""
        self.fitted_models = []  # reset fitted models
        self.check_if_data()
        # fit model for each unique value in column
        values = list(set(self.i[column]))
        i = values.index(test_set)
        train_set = values[:i] + values[i + 1:]
        test_set = [values[i]]
        x_train, y_train = self._xy_(specifics=(column, train_set))
        x_test, y_test = self._xy_(specifics=(column, test_set))
        model = clone(self.model)
        fits = model.fit(x_train, y_train)
        self.fitted_models.append((fits, x_test, y_test))

    def scores(self, scores: dict = None):
        """"""
        if not scores:
            scores = {'accuracy': accuracy_score, 'precision': precision_score,
                      'recall': recall_score, 'f1': f1_score}
        scoring = {s: [] for s in scores.keys()}
        for fitted, x_test, y_test in self.fitted_models:
            y_pred = fitted.predict(x_test)
            for name, score in scores.items():
                scoring[name].append(score(y_test, y_pred))

        scores = []
        for k, v in scoring.items():
            s = 0
            if len(self.fitted_models) > 1:
                s = round(stdev(v), 4)
                v = [mean(v)]
            v = round(v[0], 4)
            scores.append(f'{k}={v}({s})')
        print(f"{self.name} - {' '.join(scores)}")

    def roc_values(self):
        """"""
        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        aucs = []
        for fitted, x_test, y_test in self.fitted_models:
            ax = plt.gca()  # create temp axis
            viz = RocCurveDisplay.from_estimator(fitted, x_test, y_test, ax=ax)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = .0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
            ax.remove()  # remove temp axis
        return mean_fpr, tprs, aucs

    def __feature_strain_taxa__(self, features, taxa: bool = True):
        """"""
        strains_list = []
        taxa_list = []
        for f in features:
            # check if feature is from phage or bacteria
            if f.startswith('p'):
                species, column, sf = 'phage', 'phage_taxa', self.pf
            elif f.startswith('b'):
                species, column, sf = 'bacterium', 'bacterium_taxa', self.bf
            else:  # if feature is from neither append NaN
                strains_list.append('NaN')
                taxa_list.append('NaN')
                continue
            strains = sf.columns[sf.loc[f[1:]] > 0].tolist()
            if taxa and column in self.i:
                taxa = set(self.i[column][self.i[species].isin(strains)])
            else:
                taxa = ['NA']
            strains_list.append(';'.join(strains))
            taxa_list.append(';'.join(taxa))
        return strains_list, taxa_list

    def feature_importance(self,
                           strain_info: bool = True, taxa_info: bool = False):
        """"""
        features = None
        scores = None
        # get importance scores
        for fitted, x_test, _ in self.fitted_models:
            if not features or not scores:
                features = x_test.columns.tolist()
                scores = {f: [] for f in features}
            for i, f in enumerate(fitted.feature_importances_):
                scores[features[i]].append(f)
        # get mean score and its std dev
        for f in scores:
            s = scores[f]
            sd = 0
            if len(s) > 1:
                sd = stdev(s)
                s = [mean(s)]
            scores[f] = {'importance': s[0], 'stdev': sd}
        fi = pd.DataFrame(scores).T
        if strain_info or taxa_info:
            strains, taxa = self.__feature_strain_taxa__(fi.index,
                                                         taxa=taxa_info)
            if taxa_info:
                fi['taxa'] = taxa
            if strain_info:
                fi['strains'] = strains
        return fi.sort_values(by='importance', ascending=False)

    def misclassified(self, interactions: bool = True,
                      taxa: bool = False, strains: bool = False):
        """"""
        # create dict
        misclassified = {}
        ex = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        if taxa:
            pt = {t: ex.copy()
                  for t in sorted(list(set(self.i['phage_taxa'])))}
            bt = {t: ex.copy()
                  for t in sorted(list(set(self.i['bacterium_taxa'])))}
            misclassified = {**misclassified, **pt, **bt}
        if strains:
            ps = {s: ex.copy() for s in self.pf.columns}
            bs = {s: ex.copy() for s in self.bf.columns}
            misclassified = {**misclassified, **ps, **bs}
        if interactions:
            misclassified = {**misclassified,
                             **{i: ex.copy() for i in self.i.index}}
        # parse fitted models
        for fitted, x_test, y_test in self.fitted_models:
            inters = x_test.index.tolist()
            y_test = y_test.tolist()
            y_pred = fitted.predict(x_test).tolist()
            # count predictions
            for i, pred in enumerate(y_pred):
                inter_data = self.i.loc[inters[i]]
                # determine if prediction is false or true positive / negative
                key = 't'
                if pred != y_test[i]:
                    key = 'f'
                if pred == 1:
                    key = f'{key}p'
                else:
                    key = f'{key}n'
                # count prediction
                if interactions:
                    misclassified[inters[i]][key] += 1
                if taxa:
                    misclassified[inter_data['phage_taxa']][key] += 1
                    misclassified[inter_data['bacterium_taxa']][key] += 1
                if strains:
                    misclassified[inter_data['phage']][key] += 1
                    misclassified[inter_data['bacterium']][key] += 1
        return pd.DataFrame(misclassified).T

    @staticmethod
    def __shapely_scores__(shap_values, x, y, fits, feature_list):
        """"""
        sv_df = pd.DataFrame(shap_values, index=x.index, columns=x.columns)
        # order so specific features are first
        if feature_list:
            features = (feature_list +
                        list(sv_df.columns.difference(feature_list)))
            sv_df = sv_df.loc[:, features]
        mean_sv = abs(sv_df).apply(['mean']).T
        mean_sv = mean_sv.sort_values(by='mean', ascending=False)
        return sv_df.loc[:, mean_sv.index].T

    @staticmethod
    def __shapely_summary_plot__(shap_values, x, feature_list, n_display,
                                 title, output):
        """"""
        # only select specific features
        sort = True
        if feature_list:
            shap_values = pd.DataFrame(shap_values, columns=x.columns.tolist())
            shap_values = np.array(shap_values[feature_list])
            x = x[feature_list]
            sort = False
            n_display = len(x)
        # summary plot
        shap.summary_plot(shap_values, x,
                          sort=sort, max_display=n_display, show=False)
        plt.title(title)
        if output:
            plt.savefig(f'{output}.png')
        else:
            plt.show()
        plt.close()

    def shapely_values(self, feature_list: list = None,
                       n_display: int = 25, output_folder: str = None,
                       include_neg: bool = False):
        """"""
        # fit model
        x, y = self._xy_()
        model = clone(self.model)
        fits = model.fit(x, y)
        # create explainer and get shap values for positive interactions
        explainer = shap.TreeExplainer(model=fits, data=x,
                                       feature_names=x.columns.tolist())
        if include_neg is False:
            x, y = x[y == 1], y[y == 1]  # only positive interactions
        shap_vals = explainer.shap_values(x, y, check_additivity=False)[..., 1]
        # file_name
        if output_folder:
            file_name = f'{output_folder}/{self.name}_shapely'
        else:
            file_name = None
        # summary plot
        self.__shapely_summary_plot__(shap_vals, x, feature_list, n_display,
                                      f'{self.name}', file_name)
        # scores
        sv_df = self.__shapely_scores__(shap_vals, x, y, fits, feature_list)
        if output_folder:
            sv_df.to_csv(f'{file_name}.csv')
        else:
            return sv_df
        return None


def data_files(folder: str, interactions: str = '', features: str = 'R7',
               datatype: int = 0):
    interaction_file = f'{folder}/{interactions}interactions.csv'
    f_folder = f'{folder}/{features}/{features}'
    pf_file = f'{f_folder}_phages_features.csv'
    bf_file = f'{f_folder}_bacteria_features.csv'
    return {'interaction_file': interaction_file,
            'pf_file': pf_file,
            'bf_file': bf_file,
            'datatype': datatype}


def roc_plot_from_values(label, fprs, tprs, aucs, axis=None):
    """"""
    if not axis:
        plt.figure(figsize=(8, 8))
        axis = plt.gca()
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(fprs, mean_tpr)
    std_auc = np.std(aucs)
    axis.plot(fprs, mean_tpr,
              label=f'{label} '
                    f'(AUC = {round(mean_auc, 4)} ({round(std_auc, 4)}))')
    if not axis:
        axis.set(xlabel="False Positive Rate", ylabel="True Positive Rate")
        axis.legend(loc="lower right")
        plt.show()


def cross_validate_roc(folder: str, models: list,
                       k: int = 5, random_state: int = 1,
                       output: str = None,
                       feature_importance: bool = False,
                       misclassified: bool = False,
                       graph_name: str = ''):
    """
    :param folder:
    :param models: list(tuples(str, str, str, int, class))
                 ; folder, interactions, features, datatype, model
    :param k:
    :param random_state:
    :param output:
    :param feature_importance:
    :param misclassified:
    :param graph_name:
    :return:
    """
    fpr_dict = {}
    tpr_dict = {}
    auc_dict = {}
    for name, interactions, features, datatype, model in models:
        model = ModelTraining(name, model)
        model.import_data(**data_files(folder, interactions,
                                       features, datatype))
        model.cv_fit(k=k, random_state=random_state)
        model.scores()
        if output and feature_importance:
            fi = model.feature_importance(strain_info=True, taxa_info=True)
            fi.to_csv(f'{output}/{name}_feature_importance.csv')
        if output and misclassified:
            ms = model.misclassified(interactions=True,
                                     taxa=True, strains=True)
            ms.to_csv(f'{output}/{name}_misclassified.csv')
        fpr, tpr, au = model.roc_values()
        fpr_dict[name] = fpr
        tpr_dict[name] = tpr
        auc_dict[name] = au
    # create plot
    plt.figure(figsize=(8, 8))
    axis = plt.gca()
    for name, _, _, _, _ in models:
        roc_plot_from_values(label=name, axis=axis, fprs=fpr_dict[name],
                             tprs=tpr_dict[name], aucs=auc_dict[name])
    axis.set(xlabel="False Positive Rate", ylabel="True Positive Rate")
    axis.legend(loc="lower right")
    plt.title(graph_name)
    if output:
        graph_name = graph_name.replace(' ', '_')
        plt.savefig(f'{output}/{graph_name}roc.png')
    else:
        plt.show()


def train_rest_test_one(folder: str, interactions: str,
                        features: str, datatype: int = 1,
                        column: str = 'phage_taxa', model=None):
    """"""
    # create model object
    mt = ModelTraining(features, model)
    mt.import_data(
        interaction_file=f'{folder}/{interactions}interactions.csv',
        pf_file=f'{folder}/{features}/{features}_phages_features.csv',
        bf_file=f'{folder}/{features}/{features}_bacteria_features.csv',
        datatype=datatype)

    # train and combined per value in column
    column_values = list(set(mt.i[column]))
    column_values.sort()

    column_values = ['Vibrio lentus', 'Vibrio splendidus']

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        for v in column_values:
            mt.name = v
            mt.loo_fit(column=column, test_set=v)
            # mt.scores()
            mc = mt.misclassified(interactions=False, taxa=True, strains=False)
            # print(v, mc.loc[v].tolist())

            print(datatype, v, '\n', mc, '\n')


def class_weight_models(name, interactions, features, datatype,
                        threads: int = 1, seed: int = None):
    """"""
    # for name, interactions, features, datatype, model in models:
    models = []
    for i in (.01, .1, .2, .3, .4, .5, .6, .7, .8, .9, .99):
        j = round(1 - i, 2)
        data = [f'{name}({i}:{j})', interactions, features, datatype,
                RandomForestClassifier(n_jobs=threads, random_state=seed,
                                       class_weight={1: i, 0: j})]
        models.append(data)
    return models


def weight_distributions(folder, interactions, threads, seed):
    for name, features, datatype in [('HGs', 'R7', 0),
                                     ('com', 'combined', 0),
                                     ('ind', '6mer', 0),
                                     ('dif', '6mer', 1),
                                     ('dis', '6mer', 2)]:
        print(f' -- {name} --')
        mt = ModelTraining('', None)
        mt.import_data(**data_files(folder, interactions, features, datatype))
        for i in (.01, .10, .20, .30, .40, .50, .60, .70, .80, .90, .99):
            j = round(1 - i, 2)
            mt.name = f'{name}({i}:{j})'
            mt.model = RandomForestClassifier(n_jobs=threads,
                                              random_state=seed,
                                              class_weight={1: i, 0: j})
            mt.cv_fit(k=10, shuffle=True, random_state=seed)
            mt.scores()


def leavo_combo_out(folder, interactions, model):
    phage_taxa = ['sipho', 'myo', 'podo', 'auto']
    bact_taxa = ['Vibrio lentus', 'Vibrio breoganii',
                 'Vibrio cyclitrophicus', 'Vibrio splendidus',
                 'Vibrio sp. 10N1', 'Vibrio sp. F12']

    for pt in phage_taxa:
        # leave phage -, bacterium taxon combination out
        mt = ModelTraining('', model)
        mt.import_data(**data_files(folder, interactions, 'R7', 0))
        i = mt.i.copy()

        for bt in bact_taxa:
            test_i = i[(i['phage_taxa'] == pt) & (i['bacterium_taxa'] == bt)]
            train_i = i.drop(test_i.index, axis=0)
            if len(test_i[test_i['interaction'] == 1]) > 1:
                # train model
                mt.i = train_i
                x_train, y_train = mt._xy_()
                model = clone(mt.model)
                fits = model.fit(x_train, y_train)
                # get scores
                mt.i = test_i
                x_test, y_test = mt._xy_()
                y_pred = fits.predict(x_test)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore",
                                            category=UndefinedMetricWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)

                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)

                (tn, fp), (fn, tp) = confusion_matrix(y_test, y_pred)
                print(f'{pt}:{bt} - {precision}, {recall} | '
                      f'{tp}, {fn}, {fp}, {tn}')


def main():
    """"""
    try:
        threads = int(argv[1])
    except IndexError:
        threads = 1

    seed = 3
    model = RandomForestClassifier(n_jobs=threads, random_state=seed)

    folder = 'input/model_data'
    # output = 'output/misclassified'
    output = None

    interactions = ''
    # interactions = 'partial_'

    species, strain, feat = 'bacterium', '10N.286.54.E7', 'R3'
    # species, strain, feat = 'phage', '1.200.O.', 'R7

    mt = ModelTraining('test', model)
    mt.import_data(**data_files(folder, interactions, feat, 0))

    mt.i = mt.i[mt.i[species] == strain]

    sv_scores = mt.shapely_values(n_display=25, include_neg=True)[:10]

    func = f'output/top_functions/{feat}_feature_functions.csv'
    func = pd.read_csv(func, index_col=0)

    for feature, scores in sv_scores.iterrows():
        prod = func.loc[feature, 'functions'].split(';')
        print(feature, round(mean(abs(scores)), 4), len(prod), prod)


if __name__ == '__main__':
    main()
