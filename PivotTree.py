import numpy as np
import random
import numpy as np
import statistics as st
from typing import Union
import math
import time

from scipy.linalg import norm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import kmedoids

from scipy.sparse import csc_matrix
from sklearn.metrics import accuracy_score, pairwise_distances

from RuleTree import *


def pariwise_computation(X, X_pivots, metric):
  start_time = time.process_time()
  X_reduced = pairwise_distances(X, X_pivots, metric)
  end_time = time.process_time()

  time_pairwise = end_time - start_time

  return X_reduced, time_pairwise


class RandomRuleStumpForest:

    def __init__(
            self,
            max_depth = 1,
            max_nbr_nodes = 32,
            min_samples_leaf = 3,
            min_samples_split = 5,
            model_type: str = 'clf',
            allow_oblique_splits: bool = False,
            force_oblique_splits: bool = False,
            max_oblique_features: int = 2,
            prune_tree: bool = True,
            random_state: int = None,
            n_jobs: int = 1,
            verbose= False,
            bootstrap: bool = False,
            n_estimators: int = 100,
            max_features: int = 100,
            max_samples: int = 100,
            save_trees: bool = False,

    ):

        self.max_depth = max_depth
        self.max_nbr_nodes = max_nbr_nodes
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.model_type = model_type
        self.allow_oblique_splits = allow_oblique_splits
        self.force_oblique_splits = force_oblique_splits
        self.max_oblique_features = max_oblique_features
        self.prune_tree = prune_tree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        # self.processPoolExecutor = None

        self.bootstrap = bootstrap
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples
        self.save_trees = save_trees

        random.seed(self.random_state)

        self.best_estimator_ = None
        self.best_random_columns_ = None

    def fit(self, X,y):
      random_sample_seed = self.random_state
      estimators = {}
      best_id, best_estimator, best_impurity, best_random_columns = None, None, np.inf, None
      sub_matrix = None

      for id in range(self.n_estimators):

        rt = RuleTree(max_depth = 1, prune_tree = False,
                      allow_oblique_splits = self.allow_oblique_splits, force_oblique_splits = self.force_oblique_splits,
                      max_oblique_features = self.max_oblique_features, random_state = self.random_state)

        np.random.seed(random_sample_seed)
        random_columns = np.random.choice(range(X.shape[1]), size = self.max_features, replace = False)
        random_sample_seed += 1

        if self.bootstrap == False:
          rt.fit(X[: , random_columns], y)
        else:
          np.random.seed(random_sample_seed)
          random_rows = np.random.choice(range(X.shape[0]), size = self.max_samples)
          random_sample_seed +=1

          rt.fit(X[: , random_columns][random_rows], y[random_rows])

        if len(rt._node_dict) > 1:
          impurity_l = rt._node_dict[1].impurity
          impurity_r = rt._node_dict[2].impurity

          l_samples = rt._node_dict[1].samples
          r_samples = rt._node_dict[2].samples
          nbr_samples = l_samples + r_samples

          impurity_ob = l_samples / nbr_samples * impurity_l + r_samples / nbr_samples * impurity_r

          #print(id, random_columns, impurity_ob)

          if impurity_ob < best_impurity:
            best_impurity = impurity_ob
            best_estimator = rt
            best_id = id
            best_random_columns = random_columns
            sub_matrix = X[:, random_columns]

          if self.save_trees:
            estimators[id] = rt

        else:
          continue

      self.best_estimator_ = best_estimator
      self.best_random_columns_ = best_random_columns

      #print(best_id, best_estimator, best_impurity, self.best_random_columns_, sub_matrix)





class PivotTreeNode:

    def __init__(self, idx, node_id, label, parent_id, is_leaf=False, clf=None, node_l=None, node_r=None,
                 samples=None, support=None, impurity=None, is_oblique=None,
                 X_pivot_discriminative = None, y_pivot_discriminative = None,
                 discriminative_pivot_names = None, discriminative_pivot_indexes = None,
                 X_pivot_descriptive = None, y_pivot_descriptive = None,
                 descriptive_pivot_names = None,  descriptive_pivot_indexes = None, pivot_used = None):

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

        self.X_pivot_discriminative = X_pivot_discriminative
        self.y_pivot_discriminative = y_pivot_discriminative
        self.discriminative_pivot_indexes = discriminative_pivot_indexes
        self.discriminative_pivot_names = discriminative_pivot_names

        self.X_pivot_descriptive = X_pivot_descriptive
        self.y_pivot_descriptive = y_pivot_descriptive
        self.descriptive_pivot_indexes = descriptive_pivot_indexes
        self.descriptive_pivot_names = descriptive_pivot_names


        self.pivot_used = pivot_used
        
        

class PivotTree:

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
            verbose=False,
            pairwise_metric = 'euclidean',
            distance_matrix = None,
            pivot_oblique_choice_features = None, #with a value of 1 or > we fit an oblique split to extract discriminative pivots from each class

            approximation: bool = True,  #set if we must use an approximation random strategy
            n_estimators: int = 50, #used only in case of approximation
            bootstrap: bool = False,
            max_samples_approximation: int = 100,

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
        self.pairwise_metric = pairwise_metric
        self.distance_matrix = distance_matrix
        self.pivot_oblique_choice_features = pivot_oblique_choice_features
        self.approximation = approximation

        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.max_samples_approximation = max_samples_approximation
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


    def _make_leaf(self, node: PivotTreeNode):
          nbr_samples = len(node.idx)
          leaf_labels = np.array([node.label] * nbr_samples).astype(int)
          node.samples = nbr_samples
          node.support = nbr_samples / len(self.__X)
          node.is_leaf = True
          self.labels_[node.idx] = leaf_labels

    def find_medoids(self, idx_iter):

          X_medoids, y_medoids, idxs_medoids = [], [], []

          X_curr, y_curr = self.__X[idx_iter], self.__y[idx_iter]

          for class_label in set(y_curr):

            idxs = np.where(y_curr == class_label) #find elements of same class

            idx_reference = (np.array(idx_iter))[idxs]

            #idx_reference punta sempre a tutta la similarity matrix iniziale

            sub_matrix_class = (self.distance_matrix[: , idx_reference])[idx_reference] #find node distance matrix of same class elements of current node

            min_id = np.argmin(sub_matrix_class.sum(axis=0))  #find index that minimize the distance in idx_reference (it corresponds to the medoid)

            best_prot_idx = idx_reference[min_id]

            X_medoids.append(self.__X[best_prot_idx])

            y_medoids.append(self.__y[best_prot_idx])

            idxs_medoids.append(best_prot_idx)

          return X_medoids, y_medoids, idxs_medoids

    def choose_pivot_fast(self, idx_iter, random_state):

          X_pivots, y_pivots, idxs_pivots = [], [], []

          X_curr, y_curr = self.__X[idx_iter], self.__y[idx_iter] #current node samples

          for class_label in set(y_curr):

            idxs = np.where(y_curr == class_label) #get index of element of same class in the node

            idx_reference = (np.array(idx_iter))[idxs]  #get indexs of same class elements considering idx_iter

            sub_matrix_class = (self.distance_matrix[: , idx_reference])[idx_iter] #consider only the columns in idx_reference and only the rows in the current node idx_iter

            force_axis = False

            if self.pivot_oblique_choice_features != None:
              if self.approximation and (len(idx_iter) > self.max_samples_approximation):
                #print('FACCIO UNO SPLIT OBLIQUE APPROSSIMATO')
                forest_obl = RandomRuleStumpForest(max_depth = 1, n_estimators = self.n_estimators, random_state = random_state,
                                       max_features = math.ceil(math.sqrt(sub_matrix_class.shape[1])), bootstrap = self.bootstrap,
                                       max_oblique_features = self.pivot_oblique_choice_features,
                                       allow_oblique_splits = True, force_oblique_splits = True, prune_tree = False)

                forest_obl.fit(sub_matrix_class, y_curr)
                model_obl, best_random_columns_selected = forest_obl.best_estimator_, forest_obl.best_random_columns_
                olq_root_node = model_obl._node_dict[0]

                idx_reference = idx_reference[best_random_columns_selected]
                sub_matrix_class = (self.distance_matrix[: , idx_reference])[idx_iter]
                #print('PRINTO LA SUBMATRIX)')
                #print(sub_matrix_class)
                #print('update matrix after forest', sub_matrix_class.shape)

              else:
                #print('FACCIO UNO SPLIT OBLIQUE NON APPROSSIMATO')
                model_obl = RuleTree(max_depth=1, random_state = random_state,
                             allow_oblique_splits = True, force_oblique_splits = True,
                             max_oblique_features = self.pivot_oblique_choice_features, prune_tree = True)

                model_obl.fit(sub_matrix_class, y_curr) #if no variance, this will be a decision tree split
                olq_root_node = model_obl._node_dict[0]

              if olq_root_node.is_oblique == True:
                #print('faccio obliquo per scegliere pivot')
                #print('olq_root_node.is_oblique', olq_root_node.is_oblique)
                model = model_obl

              else:
                #print('faccio axis per scegliere pivot')
                #print('olq_root_node.is_oblique', olq_root_node.is_oblique)
                force_axis = True


            if (type(self.pivot_oblique_choice_features) == type(None)) or (force_axis == True):
              if self.approximation and (len(idx_iter) > self.max_samples_approximation):
                #print('FACCIO UNO SPLIT AXIS PARALLE APPROSSIMATO')
                forest_axis = RandomRuleStumpForest(max_depth = 1, n_estimators = self.n_estimators, random_state = random_state,
                                       max_features = math.ceil(math.sqrt(sub_matrix_class.shape[1])), bootstrap = self.bootstrap,
                                       max_oblique_features = self.pivot_oblique_choice_features,
                                       allow_oblique_splits = False, force_oblique_splits = False, prune_tree = False)

                forest_axis.fit(sub_matrix_class, y_curr)
                model_axis, best_random_columns_selected = forest_axis.best_estimator_, forest_axis.best_random_columns_
                model = model_axis

                idx_reference = idx_reference[best_random_columns_selected]
                sub_matrix_class = (self.distance_matrix[: , idx_reference])[idx_iter]
                #print('PRINTO LA SUBMATRIX)')
                #print(sub_matrix_class)
                #print('update matrix after forest', sub_matrix_class.shape)


              else:
                  #print('FACCIO UNO SPLIT AXIS NON APPROSSIMATO')

                  model_axis = RuleTree(max_depth=1, random_state = random_state,
                             allow_oblique_splits = False, force_oblique_splits = False,
                             max_oblique_features = self.pivot_oblique_choice_features, prune_tree = True)

                  model = model_axis

            if model == None:
              print('NO CLASSIFIER DEFINED for class from approximation process', class_label)
              continue

            model.fit(sub_matrix_class, y_curr)

            root_node = model._node_dict[0] # get root node of the decision stump
            clf = root_node.clf #get classifier stump associated to root node

            if clf == None:
              print('NO CLASSIFIER DEFINED for class', class_label)
              continue

            if not root_node.is_oblique:
              feat = clf.tree_.feature[0]
              X_prot, y_prot = self.__X[idx_reference[feat]], self.__y[idx_reference[feat]]
              #print('COMBACIANO???', class_label, y_prot)

              X_pivots.append(np.array([X_prot]))
              y_pivots.append(np.array([y_prot]))
              idxs_pivots.append(np.array([idx_reference[feat]]))

            else:
              pca_feat = clf.oblq_clf.tree_.feature[0]
              feat_list = np.where(clf.u_weights != 0)[0].tolist()

              coef = clf.householder_matrix[:, pca_feat][feat_list].tolist()

              X_prot, y_prot = self.__X[idx_reference[feat_list]], self.__y[idx_reference[feat_list]]
              #print('COMBACIANO???', class_label, y_prot)

              X_pivots.append(X_prot)
              y_pivots.append(y_prot)
              idxs_pivots.append(idx_reference[feat_list])

          if len(X_pivots) == 0: #if no pivot extracted
            return X_pivots, y_pivots, idxs_pivots

          if self.pivot_oblique_choice_features != None:
            X_pivots = list(np.concatenate(X_pivots))
            #if type(y_pivots[0]) == type(np.array([1])):
            y_pivots = list(np.concatenate(y_pivots))
            #if type(idxs_pivots[0]) == type(np.array([1])):
            idxs_pivots = list(np.concatenate(idxs_pivots))

          return X_pivots, y_pivots, idxs_pivots

    def fit(self, X, y):
        # self.processPoolExecutor = ProcessPoolExecutor(self.n_jobs, initializer=init_pool, initargs=(__X,))

        self.__X = X
        self.__y = y

        if type(self.distance_matrix) == type(None):
          #print('COMPUTING MATRIX')
          sparse_matrix = (self.__X)
          pairwise_dist_matrix = pairwise_distances(sparse_matrix, metric = self.pairwise_metric)
          self.distance_matrix = pairwise_dist_matrix
        else:
          1
          #print('PRECOMPUTED MATRIX')


        n_features = X.shape[1]
        n_idx = X.shape[0]
        idx = np.arange(n_idx)

        self.labels_ = -1 * np.ones(n_idx).astype(int)

        node_id = 0
        majority_class = st.mode(self.__y)
        root_node = PivotTreeNode(idx, node_id, majority_class, parent_id=-1) ##########
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
            
            #identify medoids

            X_medoids, y_medoids, idxs_medoids = self.find_medoids(idx_iter)

            node.X_pivot_descriptive = X_medoids
            node.y_pivot_descriptive = y_medoids
            node.descriptive_pivot_indexes = idxs_medoids
            node.descriptive_pivot_names = ['node_id: ' + str(node.node_id) + ' medoid: ' + str(x) for x in (idxs_medoids)] #name medoids!!


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

            
            #identify discriminative pivots

            X_prots, y_prots, idxs_pivots = self.choose_pivot_fast(idx_iter, random_state = self.random_state)

            if len(X_prots) == 0 and len(X_medoids) == 0:
              self._make_leaf(node)
              nbr_curr_nodes +=1
              continue

            if self.pivot_oblique_choice_features == None and len(X_prots) > 0 :
              X_prots, y_prots , idxs_pivots = np.concatenate(X_prots), np.concatenate(y_prots), np.concatenate(idxs_pivots)

            node.X_pivot_discriminative = X_prots
            node.y_pivot_discriminative = y_prots
            node.discriminative_pivot_indexes = idxs_pivots
            node.discriminative_pivot_names = ['node_id: ' + str(node.node_id) + ' pivot: ' + str(x) for x in (idxs_pivots)] #name pivots!!


            #print(idxs_pivots)
            #print(idxs_medoids)

            #print('overall_prototypes_idxs', [idxs_pivots, idxs_medoids])

            overall_prototypes_idxs = (np.concatenate([idxs_pivots, idxs_medoids])).astype(int)
            #print('overall_prototypes_idxs', overall_prototypes_idxs)

            X_reduced = (self.distance_matrix[: , overall_prototypes_idxs])[idx_iter]


            clf.fit(X_reduced, self.__y[idx_iter]) #fit decision tree
            labels = clf.apply(X_reduced)

            is_oblique = False
            if self.allow_oblique_splits:
                olq_clf = ObliqueHouseHolderSplit(
                    max_oblique_features=self.max_oblique_features,
                    min_samples_leaf=self.min_samples_leaf,
                    min_samples_split=self.min_samples_split,
                    random_state=self.random_state,

                )

                obl_fitting = olq_clf.fit(X_reduced, self.__y[idx_iter]) #fit oblique tree

                                                           #if no variance to explain here then WHAT DO?
                                                           #1)MODIFY HHCART split so that X_house = X.dot(Identity)
                                                           #and do decision split considering the original X_reduced

                                                           #2)RETURN NONE or smt and consider this a decision tree axis parallel split.

                if olq_clf.oblq_clf != None:
                  #if oblq_clf is not NONE, meaning we have successfully done the split,
                  #then perform the comparison with the splits, otherwise use the decision tree split

                  labels_ob = olq_clf.apply(X_reduced)

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

                  if self.force_oblique_splits or impurity_ob < impurity_ap:   # acc_olq > acc_par:
                      #print('OBLIQUE')
                      clf = olq_clf
                      is_oblique = True

            labels = clf.apply(X_reduced)
            y_pred = clf.predict(X_reduced)

            if len(np.unique(labels)) == 1:
                print('HO UNIQUE LABELS')
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

            node_l = PivotTreeNode(idx=idx_all_l, node_id=node_id, label=label_l,
                                  parent_id=node.node_id, impurity=impurity_l,
                                  ) ###

            node_id += 1
            if not is_oblique:
                impurity_r = clf.tree_.impurity[2]
            else:
                impurity_r = clf.oblq_clf.tree_.impurity[2]

            node_r = PivotTreeNode(idx=idx_all_r, node_id=node_id, label=label_r,
                                  parent_id=node.node_id, impurity=impurity_r,
                                  )  ###
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
            self._prune_tree()

        self.root_ = root_node

        self.label_encoder_ = LabelEncoder()
        if self.model_type == 'clf':
                self.labels_ = self.label_encoder_.fit_transform(self.labels_)

        if len(self._node_dict) > 1:
          self.rules_to_tree_print_ = self._get_rules_to_print_tree()
          self.rules_ = self._get_rules()
          self.rules_ = self._compact_rules()
          self.rules_s_ = self._rules2str()


    def predict(self, X, get_leaf=False, get_rule=False):
        idx = np.arange(X.shape[0])
        labels, leaves = self._predict(X, idx, self.root_)
        if self.model_type == 'clf':
            labels = self.label_encoder_.transform(labels)

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

            clf = node.clf #classifier/split trained on the pivots in the node (associated with node's pivots)

            X_prots = node.X_pivot_discriminative #get discriminative pivots

            X_medoids = node.X_pivot_descriptive #get descriptive pivots

            if len(X_prots) == 0:
              overall_pivots = X_medoids
            else:
              overall_pivots = np.concatenate([X_prots, X_medoids])

            X_reduced = pairwise_distances(X[idx_iter], overall_pivots,  metric = self.pairwise_metric)

            labels = clf.apply(X_reduced)
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
                idxs_pivots = np.concatenate([next_node.discriminative_pivot_indexes, next_node.descriptive_pivot_indexes])
                name_temp = 'node_id: ' + str(next_node.node_id) + '  '

                # print(next_node.node_id, type(next_node.clf), next)
                if not next_node.is_oblique:
                    feat = clf.tree_.feature[0] #find index feature
                    thr = clf.tree_.threshold[0]
                    cat = feat in self.cat_indexes
                    coef = None

                    feat = idxs_pivots[feat]   #find index corresponding to feature

                    if feat in next_node.discriminative_pivot_indexes:
                      feat = name_temp + 'pivot: ' + str(feat)
                    if feat in next_node.descriptive_pivot_indexes:
                      feat = name_temp + 'medoid: ' + str(feat)

                    #print('i split according to: ', feat)

                    next_node.pivot_used = [feat]  #we specify the pivot used for the split

                else:
                    pca_feat = clf.oblq_clf.tree_.feature[0]
                    thr = clf.oblq_clf.tree_.threshold[0]
                    feat = np.where(clf.u_weights != 0)[0].tolist() #find index features
                    coef = clf.householder_matrix[:, pca_feat][feat].tolist()
                    feat = [idxs_pivots[id] for id in feat] #find index corresponding to features

                    feat = [name_temp + 'pivot: ' + str(f) if (f in next_node.discriminative_pivot_indexes)
                    else name_temp + 'medoid: ' + str(f) for f in feat]
                    cat = False

                    next_node.pivot_used = [feat] ###we specify the pivots used for the split


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

    def print_tree(self, precision=2, cat_precision=0):
        #nbr_features = self.__X.shape[1]
        rules = self.rules_to_tree_print_
        print(rules)
       # if self.feature_names is None:
       #     self.feature_names = ["__X%s" % i for i in range(nbr_features)]

        s_rules = ""
        for rule in rules:

            is_rule = rule[0]
            depth = rule[-1]
            ident = "  " * depth
            if is_rule:

                _, feat_list, coef_list, thr, cat, _ = rule
                if len(feat_list) == 1:
                    feat_s = "%s" % feat_list[0]
                else:
                    feat_s = [
                        "%s %s"
                        % (np.round(coef_list[i], precision), feat_list[i])
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

    def _rules2str(self):
        self.rules_s_ = dict()

        for leaf_id in self.rules_:
            rule = self.rules_[leaf_id]
            self.rules_s_[leaf_id] = list()
            for cond in rule:
               # print(cond)
                if len(cond) == 1:
                    cond_s = ('%s' % cond[0])
                else:
                    feat, symb, thr, cat, is_oblique, coef = cond

                    if not is_oblique:
                        feat_s = "%s" % [feat]
                    else:
                        feat_s = [
                            "%s %s"
                            % (np.round(coef[i], self.precision), [feat[i]])
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


    def _get_rules_to_print_tree(self):
        idx = np.arange(self.__X.shape[0])
        return self.__get_rules_to_print_tree(idx, self.root_, 0)

    def __get_rules_to_print_tree(self, idx_iter, node: PivotTreeNode, cur_depth):
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

            #print('QUI', [node.discriminative_pivot_indexes, node.descriptive_pivot_indexes])
            idxs_pivots = np.concatenate([node.discriminative_pivot_indexes, node.descriptive_pivot_indexes])

            X_prots = node.X_pivot_discriminative #get discriminative pivots
            X_medoids = node.X_pivot_descriptive #get descriptive pivots

            if len(X_prots) == 0:
              overall_pivots = X_medoids
            else:
              overall_pivots = np.concatenate([X_prots, X_medoids])

            name_temp = 'node_id: ' + str(node.node_id) + '  '

            X_reduced = pairwise_distances(self.__X[idx_iter], overall_pivots ,  metric='euclidean')

            labels = clf.apply(X_reduced)

            idx_l, idx_r = np.where(labels == 1)[0], np.where(labels == 2)[0]
            idx_all_l = idx_iter[idx_l]
            idx_all_r = idx_iter[idx_r]


            if not node.is_oblique:
                feat = clf.tree_.feature[0]
                thr = clf.tree_.threshold[0]
                cat = feat in self.cat_indexes
                feat = idxs_pivots[feat]

                if feat in node.discriminative_pivot_indexes:
                  feat = name_temp + 'pivot: ' + str(feat)
                if feat in node.descriptive_pivot_indexes:
                  feat = name_temp + 'medoid: ' + str(feat)

                rule = (True, [feat], [1.0], thr, cat, cur_depth)

            else:
                pca_feat = clf.oblq_clf.tree_.feature[0]
                thr = clf.oblq_clf.tree_.threshold[0]
                feat_list = np.where(clf.u_weights != 0)[0].tolist()
                coef = clf.householder_matrix[:, pca_feat][feat_list].tolist()
                # coef = StandardScaler().inverse_transform(coef)

                feat_list = idxs_pivots[feat_list]
                feat_list = [name_temp + 'pivot: ' + str(f) if (f in node.discriminative_pivot_indexes)
                    else name_temp + 'medoid: ' + str(f) for f in feat_list]

                rule = (True, feat_list, coef, thr, False, cur_depth)

            rules.append(rule)
            rules += self.__get_rules_to_print_tree(idx_all_l, node.node_l, cur_depth + 1)
            rules += self.__get_rules_to_print_tree(idx_all_r, node.node_r, cur_depth + 1)
            return rules






