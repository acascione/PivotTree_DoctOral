import random
import numpy as np
import statistics as st
from typing import Union

from scipy.linalg import norm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def prepare_data(X_original, max_nbr_values, max_nbr_values_cat):
    X = np.copy(X_original)
    feature_values = dict()
    n_features = X.shape[1]
    is_categorical_feature = np.full_like(np.zeros(n_features, dtype=bool), False)
    for feature in range(n_features):
        values = np.unique(X[:, feature])
        vals = None
        if len(values) > max_nbr_values:
            _, vals = np.histogram(values, bins=max_nbr_values)
            values = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]
        feature_values[feature] = values

        if len(values) >= max_nbr_values_cat:
            is_categorical_feature[feature] = True

            if vals is not None:
                for original_val_idx in range(X.shape[0]):
                    for min_val, max_val, binned_val in zip(vals[:-1], vals[1:], values):
                        original_val = X[original_val_idx, feature]
                        if min_val < original_val < max_val:
                            X[original_val_idx, feature] = binned_val
                            break

    return is_categorical_feature, X


def is_verified_x_r_(x, rule, count_cond=False, relative_count=False):
    verified = True
    count_verified = 0
    for cond in rule[:-1]:
        feat, symb, thr, cat, is_oblique, coef = cond

        if not is_oblique:
            if cat:  # categorical
                if symb == '=':
                    verified = verified and x[feat] == thr
                else:  # symb = '!='
                    verified = verified and x[feat] != thr

            else:  # continuous
                if symb == '<=':
                    verified = verified and x[feat] <= thr
                else:  # symb = '>'
                    verified = verified and x[feat] > thr
        else:  # is oblique
            if symb == '<=':
                verified = verified and np.sum(x[feat] * coef) <= thr
            else:  # symb = '>'
                verified = verified and np.sum(x[feat] * coef) > thr

        if verified:
            count_verified += 1

        if not verified and not count_cond:
            return False

    if count_cond:

        if relative_count:
            return count_verified / len(rule[:-1])

        return count_verified

    return verified


def is_verified_r_(X, rule, count_cond=False, relative_count=False):
    verified_list = list()
    for x in X:
        verified_list.append(is_verified_x_r_(x, rule, count_cond, relative_count))

    verified_list = np.array(verified_list)
    return verified_list


def is_verified(X, rules, count_cond=False, relative_count=False):
    verified_list = list()
    for leaf, rule in rules.items():
        verified_list.append(is_verified_r_(X, rule, count_cond, relative_count))

    verified_list = np.array(verified_list).T
    return verified_list





class ObliqueHouseHolderSplit:
    def __init__(
        self,
        pca=None,
        max_oblique_features=2,
        min_samples_leaf=3,
        min_samples_split=5,
        tau=1e-4,
        model_type='clf',
        random_state=None,
    ):
        self.pca = pca
        self.max_oblique_features = max_oblique_features
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.tau = tau
        self.model_type = model_type
        self.dominant_ev = None
        self.u_weights = None
        self.householder_matrix = None
        self.oblq_clf = None
        self.random_state = random_state

    def transform(self, X):
        X_house = X.dot(self.householder_matrix)
        return X_house

    def fit(self, X, y):

        n_features = X.shape[1]

        self.pca = PCA(n_components=1)
        self.pca.fit(X)

        self.dominant_ev = self.pca.components_[0]
        I = np.diag(np.ones(n_features))

        diff_w_means = np.sqrt(((I - self.dominant_ev) ** 2).sum(axis=1))

        if (diff_w_means > self.tau).sum() == 0:
            print("No variance to explain.")
            return 'no_variance'

        idx_max_diff = np.argmax(diff_w_means)
        e_vector = np.zeros(n_features)
        e_vector[idx_max_diff] = 1.0
        self.u_weights = (e_vector - self.dominant_ev) / norm(e_vector - self.dominant_ev)

        if self.max_oblique_features < n_features:
            idx_w = np.argpartition(np.abs(self.u_weights), -self.max_oblique_features)[-self.max_oblique_features:]
            u_weights_new = np.zeros(n_features)
            u_weights_new[idx_w] = self.u_weights[idx_w]
            self.u_weights = u_weights_new

        if (diff_w_means > self.tau).sum() == 0:  ### new section, if no variance to explain (an axis parallel split is sufficient)
            #print("No variance to explain.")     ### keep the original X (do X.dot(I)), so the self.householder_matrix can be I
            self.householder_matrix = I
        else:
            self.householder_matrix = I - 2 * self.u_weights[:, np.newaxis].dot(self.u_weights[:, np.newaxis].T)

        X_house = self.transform(X)

        if self.model_type == 'clf':
            self.oblq_clf = DecisionTreeClassifier(
                max_depth=1,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
            )
        elif self.model_type == 'reg':
            self.oblq_clf = DecisionTreeRegressor(
                max_depth=1,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
            )
        else:
            raise Exception('Unknown model %s' % self.model_type)
        self.oblq_clf.fit(X_house, y)

    def predict(self, X):
        X_house = self.transform(X)
        return self.oblq_clf.predict(X_house)

    def apply(self, X):
        X_house = self.transform(X)
        return self.oblq_clf.apply(X_house)
    
    
    
    
# PROBLEMA: può capitare che la feature venga binnizzata e poi dichiarata categorica.
# In questo caso il programma genera dei threshold che non esistono nei dati
# che causa un problema in quanto il confronto è esatto (==)

class RuleTreeNode:

    def __init__(self, idx, node_id, label, parent_id, is_leaf=False, clf=None, node_l=None, node_r=None,
                 samples=None, support=None, impurity=None, is_oblique=None):
        self.idx = idx
        self.node_id = node_id
        self.label = label
        self.is_leaf = is_leaf
        self.clf = clf
        self.node_l = node_l
        self.node_r = node_r
        self.samples = samples
        self.support = support
        self.impurity = impurity
        self.is_oblique = is_oblique
        self.parent_id = parent_id


class RuleTree:

    def __init__(
            self,
            max_depth: int = 4,
            max_nbr_nodes: int = 32,
            min_samples_leaf: int = 3,
            min_samples_split: int = 5,
            max_nbr_values: Union[int, float] = np.inf,
            max_nbr_values_cat: Union[int, float] = np.inf,
            model_type: str = 'clf',
            allow_oblique_splits: bool = False,
            force_oblique_splits: bool = False,
            max_oblique_features: int = 2,
            prune_tree: bool = True,
            feature_names: list = None,
            precision: int = 2,
            cat_precision: int = 2,
            random_state: int = None,
            n_jobs: int = 1,
            verbose=False
    ):
        self.max_depth = max_depth
        self.max_nbr_nodes = max_nbr_nodes
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_nbr_values = max_nbr_values
        self.max_nbr_values_cat = max_nbr_values_cat
        self.model_type = model_type
        self.allow_oblique_splits = allow_oblique_splits
        self.force_oblique_splits = force_oblique_splits
        self.max_oblique_features = max_oblique_features
        self.prune_tree = prune_tree
        self.feature_names = feature_names
        self.precision = precision
        self.cat_precision = cat_precision
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        # self.processPoolExecutor = None

        random.seed(self.random_state)

        self.__X = None
        self.__y = None
        self.labels_ = None
        self.root_ = None
        self.label_encoder_ = None
        self.cat_indexes = None
        self.con_indexes = None
        self.is_categorical_feature = None
        self.__queue = list()
        self.__tree_structure = dict()
        self._node_dict = dict()
        self.__leaf_rule = dict()
        self.rules_to_tree_print_ = None
        self.rules_ = None
        self.rules_s_ = None

    def _make_leaf(self, node: RuleTreeNode):
        nbr_samples = len(node.idx)
        leaf_labels = np.array([node.label] * nbr_samples).astype(int)
        node.samples = nbr_samples
        node.support = nbr_samples / len(self.__X)
        node.is_leaf = True
        self.labels_[node.idx] = leaf_labels

    def fit(self, X, y):
        # self.processPoolExecutor = ProcessPoolExecutor(self.n_jobs, initializer=init_pool, initargs=(__X,))

        if self.feature_names is None:
            self.feature_names = np.array(['feature %s' % i for i in range(X.shape[1])])

        self.__X = X
        self.__y = y

        n_features = X.shape[1]
        n_idx = X.shape[0]
        idx = np.arange(n_idx)

        self.labels_ = -1 * np.ones(n_idx).astype(int)

        node_id = 0
        majority_class = st.mode(self.__y)
        root_node = RuleTreeNode(idx, node_id, majority_class, parent_id=-1)
        self._node_dict[root_node.node_id] = root_node

        self.__queue.append((idx, 0, root_node))

        nbr_curr_nodes = 0

        res = prepare_data(X, self.max_nbr_values, self.max_nbr_values_cat)
        self.is_categorical_feature, X = res

        self.__X = X

        self.con_indexes = np.array([i for i in range(n_features) if not self.is_categorical_feature[i]])
        self.cat_indexes = np.array([i for i in range(n_features) if self.is_categorical_feature[i]])

        while len(self.__queue) > 0 and nbr_curr_nodes + len(self.__queue) <= self.max_nbr_nodes:
            (idx_iter, node_depth, node) = self.__queue.pop(0)

            if len(np.unique(self.__y[idx_iter])) == 1:
                self._make_leaf(node)
                nbr_curr_nodes += 1
                continue

            nbr_samples = len(idx_iter)

            if nbr_curr_nodes + len(self.__queue) + 1 >= self.max_nbr_nodes \
                    or nbr_samples < self.min_samples_split \
                    or node_depth >= self.max_depth:
                self._make_leaf(node)
                nbr_curr_nodes += 1
                continue

            if self.model_type == 'clf':
                clf = DecisionTreeClassifier(
                    max_depth=1,
                    min_samples_leaf=self.min_samples_leaf,
                    min_samples_split=self.min_samples_split,
                    random_state=self.random_state,
                )
            elif self.model_type == 'reg':
                clf = DecisionTreeRegressor(
                    max_depth=1,
                    min_samples_leaf=self.min_samples_leaf,
                    min_samples_split=self.min_samples_split,
                    random_state=self.random_state,
                )
            else:
                raise Exception('Unknown model %s' % self.model_type)

            clf.fit(self.__X[idx_iter], self.__y[idx_iter])
            labels = clf.apply(self.__X[idx_iter])
            # y_pred = clf.predict(self.__X[idx_iter])

            is_oblique = False
            if self.allow_oblique_splits:
                olq_clf = ObliqueHouseHolderSplit(
                    max_oblique_features=self.max_oblique_features,
                    min_samples_leaf=self.min_samples_leaf,
                    min_samples_split=self.min_samples_split,
                    random_state=self.random_state,
                )

                olq_fit_result = olq_clf.fit(self.__X[idx_iter], self.__y[idx_iter]) #if there is no variance, i return 'no_variance'

                if olq_fit_result == 'no_variance':  #if no variance, use the axis parallel
                  olq_clf = clf
                else: #else, check the impurities
                  labels_ob = olq_clf.apply(self.__X[idx_iter])

                  # y_pred_ob = olq_clf.predict(self.__X[idx_iter])

                  vals, counts = np.unique(labels, return_counts=True)
                  if len(vals) == 1:
                      impurity_ap = np.inf
                  else:
                      impurity_l = clf.tree_.impurity[1]
                      impurity_r = clf.tree_.impurity[2]
                      impurity_ap = counts[0] / nbr_samples * impurity_l + counts[1] / nbr_samples * impurity_r

                  vals, counts = np.unique(labels_ob, return_counts=True)
                  if len(vals) == 1:
                      impurity_ob = np.inf
                  else:
                      impurity_l_ob = olq_clf.oblq_clf.tree_.impurity[1]
                      impurity_r_ob = olq_clf.oblq_clf.tree_.impurity[2]
                      impurity_ob = counts[0] / nbr_samples * impurity_l_ob + counts[1] / nbr_samples * impurity_r_ob

                # acc_par = accuracy_score(self.__y[idx_iter], y_pred)
                # acc_olq = accuracy_score(self.__y[idx_iter], y_pred_ob)

                if self.force_oblique_splits or impurity_ob < impurity_ap:   # acc_olq > acc_par:
                    clf = olq_clf
                    is_oblique = True

                    if olq_fit_result == 'no_variance':
                      is_oblique = False

            #labels = clf.apply(self.__X[idx_iter])
            #y_pred = clf.predict(self.__X[idx_iter])
            #if len(np.unique(labels)) == 1 or len(np.unique(y_pred)) == 1:
            #    self._make_leaf(node)
            #    nbr_curr_nodes += 1
            #    continue

            labels = clf.apply(self.__X[idx_iter])
            y_pred = clf.predict(self.__X[idx_iter])

            if len(np.unique(labels)) == 1:
                self._make_leaf(node)
                nbr_curr_nodes += 1
                continue

            idx_l, idx_r = np.where(labels == 1)[0], np.where(labels == 2)[0]

            idx_all_l = idx_iter[idx_l]
            idx_all_r = idx_iter[idx_r]

            label_l = st.mode(self.__y[idx_all_l])
            label_r = st.mode(self.__y[idx_all_r])

            node_id += 1
            if not is_oblique:
                impurity_l = clf.tree_.impurity[1]
            else:
                impurity_l = clf.oblq_clf.tree_.impurity[1]
            node_l = RuleTreeNode(idx=idx_all_l, node_id=node_id, label=label_l,
                                  parent_id=node.node_id, impurity=impurity_l)

            node_id += 1
            if not is_oblique:
                impurity_r = clf.tree_.impurity[2]
            else:
                impurity_r = clf.oblq_clf.tree_.impurity[2]
            node_r = RuleTreeNode(idx=idx_all_r, node_id=node_id, label=label_r,
                                  parent_id=node.node_id, impurity=impurity_r)

            node.clf = clf
            node.node_l = node_l
            node.node_r = node_r
            if not is_oblique:
                node.impurity = clf.tree_.impurity[0]
            else:
                node.impurity = clf.oblq_clf.tree_.impurity[0]
            node.is_oblique = is_oblique

            self.__tree_structure[node.node_id] = (node_l.node_id, node_r.node_id)
            self._node_dict[node_l.node_id] = node_l
            self._node_dict[node_r.node_id] = node_r

            self.__queue.append((idx_all_l, node_depth + 1, node_l))
            self.__queue.append((idx_all_r, node_depth + 1, node_r))

        if self.prune_tree:
            if self.max_depth > 1:
              self._prune_tree()

        self.root_ = root_node

        self.label_encoder_ = LabelEncoder()
        if self.model_type == 'clf':
           # print('le labels non trasformate', self.labels_)
           # print(set(self.labels_))
            self.labels_ = self.label_encoder_.fit_transform(self.labels_)
           # print('le labels TRASFORMATE', self.labels_)
           # print(set(self.labels_))


        if len(self._node_dict) > 1:
          self.rules_to_tree_print_ = self._get_rules_to_print_tree()
          self.rules_ = self._get_rules()
          self.rules_ = self._compact_rules()
          self.rules_s_ = self._rules2str()

    def predict(self, X, get_leaf=False, get_rule=False):
        idx = np.arange(X.shape[0])
        labels, leaves = self._predict(X, idx, self.root_)
        if self.model_type == 'clf':
            print(labels)
            labels = self.label_encoder_.transform(labels)
            print(labels)

        if get_leaf and get_rule:
            rules = list()
            for leaf_id in leaves:
                rules.append(self.rules_s_[leaf_id])
            rules = np.array(rules)
            return labels, leaves, rules

        if get_rule:
            rules = list()
            for leaf_id in leaves:
                rules.append(self.rules_s_[leaf_id])
            rules = np.array(rules)
            return labels, rules

        if get_leaf:
            return labels, leaves

        return labels

    def _predict(self, X, idx, node):
        idx_iter = idx

        if node.is_leaf:
            labels = np.array([node.label] * len(idx_iter))
            leaves = np.array([node.node_id] * len(idx_iter)).astype(int)

            return labels, leaves

        else:

            clf = node.clf
            labels = clf.apply(X[idx_iter])
            leaves = np.zeros(len(labels)).astype(int)

            idx_l, idx_r = np.where(labels == 1)[0], np.where(labels == 2)[0]
            idx_all_l = idx_iter[idx_l]
            idx_all_r = idx_iter[idx_r]

            if len(idx_all_l) > 0:
                labels_l, leaves_l = self._predict(X, idx_all_l, node.node_l)
                labels[idx_l] = labels_l
                leaves[idx_l] = leaves_l

            if len(idx_all_r) > 0:
                labels_r, leaves_r = self._predict(X, idx_all_r, node.node_r)
                labels[idx_r] = labels_r
                leaves[idx_r] = leaves_r

            return labels, leaves

    def get_axes2d(self, eps=1, X=None):
        idx = np.arange(self.__X.shape[0])

        if X is None:
            X = self.__X

        return self._get_axes2d(idx, self.root_, eps, X)

    def _get_axes2d(self, idx, node: RuleTreeNode, eps, X):
        idx_iter = idx

        axes2d = list()

        if node.is_leaf:
            return []

        else:
            clf = node.clf
            labels = clf.apply(self.__X[idx_iter])

            idx_l, idx_r = np.where(labels == 1)[0], np.where(labels == 2)[0]
            idx_all_l = idx_iter[idx_l]
            idx_all_r = idx_iter[idx_r]

            x_min, x_max = X[idx_iter][:, 0].min(), X[idx_iter][:, 0].max()
            y_min, y_max = X[idx_iter][:, 1].min(), X[idx_iter][:, 1].max()

            if not node.is_oblique:
                feat = clf.tree_.feature[0]
                thr = clf.tree_.threshold[0]

                if feat == 0:
                    axes = [[thr, thr], [y_min - eps, y_max + eps]]
                else:
                    axes = [[x_min - eps, x_max + eps], [thr, thr]]
            else:
                def line_fun(x):
                    f = clf.oblq_clf.tree_.feature[0]
                    b = clf.oblq_clf.tree_.threshold[0]
                    m = clf.householder_matrix[:, f][0] / clf.householder_matrix[:, f][1]
                    y = b / clf.householder_matrix[:, f][1] - m * x
                    return y

                axes = [
                    [x_min - eps, x_max + eps],
                    [line_fun(x_min - eps), line_fun(x_max + eps)],
                ]

            axes2d.append(axes)

            axes2d += self._get_axes2d(idx_all_l, node.node_l, eps, X)
            axes2d += self._get_axes2d(idx_all_r, node.node_r, eps, X)

            return axes2d

    def _get_rules_to_print_tree(self):
        idx = np.arange(self.__X.shape[0])
        return self.__get_rules_to_print_tree(idx, self.root_, 0)

    def __get_rules_to_print_tree(self, idx_iter, node: RuleTreeNode, cur_depth):
        rules = list()

        if node.is_leaf:
            label = node.label
            if self.model_type == 'clf':
                label = self.label_encoder_.transform([node.label])[0]
            leaf = (False, label, node.samples, node.support, node.node_id, cur_depth)

            rules.append(leaf)
            return rules

        else:
            clf = node.clf
            labels = clf.apply(self.__X[idx_iter])

            idx_l, idx_r = np.where(labels == 1)[0], np.where(labels == 2)[0]
            idx_all_l = idx_iter[idx_l]
            idx_all_r = idx_iter[idx_r]

            if not node.is_oblique:
                feat = clf.tree_.feature[0]
                thr = clf.tree_.threshold[0]
                cat = feat in self.cat_indexes
                rule = (True, [feat], [1.0], thr, cat, cur_depth)
            else:
                pca_feat = clf.oblq_clf.tree_.feature[0]
                thr = clf.oblq_clf.tree_.threshold[0]
                feat_list = np.where(clf.u_weights != 0)[0].tolist()
                coef = clf.householder_matrix[:, pca_feat][feat_list].tolist()
                # coef = StandardScaler().inverse_transform(coef)
                rule = (True, feat_list, coef, thr, False, cur_depth)

            rules.append(rule)
            rules += self.__get_rules_to_print_tree(idx_all_l, node.node_l, cur_depth + 1)
            rules += self.__get_rules_to_print_tree(idx_all_r, node.node_r, cur_depth + 1)
            return rules

    def _prune_tree(self):

         while True:

             tree_pruned = False
             nodes_to_remove = list()
             for node_id in self.__tree_structure:
                 node_l, node_r = self.__tree_structure[node_id]
                 if self._node_dict[node_l].is_leaf and self._node_dict[node_r].is_leaf and self._node_dict[node_l].label == \
                         self._node_dict[node_r].label:
                     self._make_leaf(self._node_dict[node_id])
                     del self._node_dict[node_l]
                     del self._node_dict[node_r]
                     self._node_dict[node_id].node_l = None
                     self._node_dict[node_id].node_r = None
                     nodes_to_remove.append(node_id)
                     tree_pruned = True

             if not tree_pruned:
                 break

             for node_id in nodes_to_remove:
                 del self.__tree_structure[node_id]

    def _get_rules(self):

        for node_id in self._node_dict:
            node = self._node_dict[node_id]
            if not node.is_leaf:
                continue

            label = node.label
            if self.model_type == 'clf':
                label = self.label_encoder_.transform([node.label])[0]
            rule = [(label,)]

            past_node_id = node_id
            next_node_id = node.parent_id
            next_node = self._node_dict[next_node_id]
            while True:

                clf = next_node.clf
                # print(next_node.node_id, type(next_node.clf), next)
                if not next_node.is_oblique:
                    feat = clf.tree_.feature[0]
                    thr = clf.tree_.threshold[0]
                    cat = feat in self.cat_indexes
                    coef = None
                else:
                    pca_feat = clf.oblq_clf.tree_.feature[0]
                    thr = clf.oblq_clf.tree_.threshold[0]
                    feat = np.where(clf.u_weights != 0)[0].tolist()
                    coef = clf.householder_matrix[:, pca_feat][feat].tolist()
                    cat = False

                if next_node.node_l.node_id == past_node_id:
                    symb = '<=' if not cat else '='
                else:
                    symb = '>' if not cat else '!='

                cond = (feat, symb, thr, cat, next_node.is_oblique, coef)

                rule.insert(0, cond)
                past_node_id = next_node_id
                next_node_id = next_node.parent_id
                self.__leaf_rule[node_id] = rule

                if next_node_id == -1:
                    break
                next_node = self._node_dict[next_node_id]

        return self.__leaf_rule

    def _compact_rules(self):
        compact_rules = dict()

        for leaf_id in self.rules_:
            rule = self.rules_[leaf_id]
            compact_rules[leaf_id] = list()
            rule_dict = dict()
            rule_ob_split = list()
            for cond in rule:
                if len(cond) > 1:
                    feat, symb, thr, cat, is_oblique, coef = cond
                    if is_oblique:
                        rule_ob_split.append(cond)
                        continue
                    if (feat, symb) not in rule_dict:
                        rule_dict[(feat, symb)] = list()
                    rule_dict[(feat, symb)].append(cond)

            for k in rule_dict:
                feat, symb = k
                cond_list = rule_dict[(feat, symb)]
                if len(cond_list) == 1:
                    cond = cond_list[0]
                    compact_rules[leaf_id].append(cond)
                else:
                    thr_list = [cond[2] for cond in cond_list]
                    cond = list(rule_dict[k][0])
                    if symb == '<=':
                        cond[2] = min(thr_list)
                    elif symb == '>':
                        cond[2] = max(thr_list)

                    compact_rules[leaf_id].append(tuple(cond))

            for cond in rule_ob_split:
                compact_rules[leaf_id].append(cond)

            compact_rules[leaf_id].append(rule[-1])

        return compact_rules

    def _rules2str(self):
        self.rules_s_ = dict()

        for leaf_id in self.rules_:
            rule = self.rules_[leaf_id]
            self.rules_s_[leaf_id] = list()
            for cond in rule:
                if len(cond) == 1:
                    cond_s = ('%s' % cond[0])
                else:
                    feat, symb, thr, cat, is_oblique, coef = cond

                    if not is_oblique:
                        feat_s = "%s" % self.feature_names[feat]
                    else:
                        feat_s = [
                            "%s %s"
                            % (np.round(coef[i], self.precision), self.feature_names[feat[i]])
                            for i in range(len(feat))
                        ]
                        feat_s = " + ".join(feat_s)

                    if not cat:
                        cond_s = "%s %s %s" % (feat_s, symb, np.round(thr, self.precision))
                    else:
                        cond_s = "%s %s %s" % (feat_s, symb, np.round(thr, self.cat_precision))
                        if self.cat_precision == 0:
                            cond_s = cond_s.replace(".0", "")
                self.rules_s_[leaf_id].append(cond_s)
            antecedent = ' & '.join(self.rules_s_[leaf_id][:-1])
            consequent = ' --> %s' % self.rules_s_[leaf_id][-1]
            self.rules_s_[leaf_id] = antecedent + consequent
        return self.rules_s_

    def print_tree(self, precision=2, cat_precision=0):
        nbr_features = self.__X.shape[1]
        rules = self.rules_to_tree_print_
        if self.feature_names is None:
            self.feature_names = ["__X%s" % i for i in range(nbr_features)]

        s_rules = ""
        for rule in rules:
            is_rule = rule[0]
            depth = rule[-1]
            ident = "  " * depth
            if is_rule:
                _, feat_list, coef_list, thr, cat, _ = rule
                if len(feat_list) == 1:
                    feat_s = "%s" % self.feature_names[feat_list[0]]
                else:
                    feat_s = [
                        "%s %s"
                        % (np.round(coef_list[i], precision), self.feature_names[feat_list[i]])
                        for i in range(len(feat_list))
                    ]
                    feat_s = " + ".join(feat_s)
                if not cat:
                    cond_s = "%s <= %s" % (feat_s, np.round(thr, precision))
                else:
                    cond_s = "%s = %s" % (feat_s, np.round(thr, cat_precision))
                    if cat_precision == 0:
                        cond_s = cond_s.replace(".0", "")
                s = "%s|-+ if %s:" % (ident, cond_s)
            else:
                _, label, samples, support, node_id, _ = rule
                support = np.round(support, precision)
                s = "%s|--> label: %s (%s, %s)" % (ident, label, samples, support)
            s_rules += "%s\n" % s

        return s_rules









































