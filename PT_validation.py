from RuleTree import *
from PivotTree import *

import sys

import pandas as pd
from sklearn.model_selection import train_test_split
from itertools import product
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score, roc_auc_score, balanced_accuracy_score, f1_score, accuracy_score, recall_score, precision_score

def generate_param_configurations(configurations):
    keys, values_list = zip(*configurations.items())
    param_combinations = list(product(*values_list))

    result_configurations = []
    for combination in param_combinations:
        config_dict = dict(zip(keys, combination))
        result_configurations.append(config_dict)

    return result_configurations


# Default values
data_set_name = ''
mode_of_evaluation = 'validation'  # or 'testing'
approximation = True
metric = 'euclidean'
time_partition = 0.0


def parse_arguments(args):
    global data_set_name, approximation, mode_of_evaluation
    
    # Parsing command line arguments
    if len(args) >= 2:
        data_set_name = args[1]
    if len(args) >= 3:
        approximation = args[2].lower() == 'true'
    if len(args) >= 4:
        mode_of_evaluation = args[3].lower()

parse_arguments(sys.argv)

#get dataset name

data_set_name

data_set_name = (data_set_name.split('/')[1])

print(data_set_name)


if 'scaled' in data_set_name:
    training_folder = 'datasets_stz_standardized/splits/training_sets/train_'
    validation_folder = 'datasets_stz_standardized/splits/validation_sets/validation_'
    test_folder = 'datasets_stz_standardized/splits/test_sets/test_'
else:
    training_folder = 'datasets_stz/splits/training_sets/train_'
    validation_folder = 'datasets_stz/splits/validation_sets/validation_'
    test_folder = 'datasets_stz/splits/test_sets/test_'

print(training_folder + data_set_name)

ds_train = pd.read_csv(training_folder + data_set_name)
ds_validation = pd.read_csv(validation_folder + data_set_name)
ds_test = pd.read_csv(test_folder + data_set_name)

class_name = ds_train.columns[-1]


X_train_inner = np.array(ds_train.loc[:, ds_train.columns != class_name].values)
y_train_inner = np.array(ds_train[class_name])

X_validation = np.array(ds_validation.loc[:, ds_validation.columns != class_name].values)
y_validation = np.array(ds_validation[class_name])

X_test = np.array(ds_test.loc[:, ds_test.columns != class_name].values)
y_test = np.array(ds_test[class_name])


if mode_of_evaluation == 'testing':
    print('Merge train with validation for testing')
    X_train_inner = np.array([x for x in X_train_inner] + [x for x in X_validation])
    y_train_inner = np.array([x for x in y_train_inner] + [x for x in y_validation])

    X_validation = X_test
    y_validation = y_test

print('Train/Val/Test size: ', len(y_train_inner), len(y_validation), len(y_test))

if mode_of_evaluation not in ['testing', 'validation']:
    raise ValueError('Shoulde be testing or validation')
    


data_set_name = 'datasets_stz/' + data_set_name


start_pairwise = time.process_time()
X_train_inner_pairwise_matrix = pairwise_distances(X_train_inner, metric = metric)
end_pairwise = time.process_time()

time_pairwise_computation = end_pairwise - start_pairwise


#pivot Tree parameters that are relevant for us
configurations = {'max_depth': [2,3,4],
                  'allow_oblique_splits' : [False],
                  'force_oblique_splits': [False],
                  'max_oblique_features' : [2],
                  'pivot_oblique_choice_features' : [None,2,3,4],
                  'approximation' : [approximation]
                 }

possible_configurations = [x for x in generate_param_configurations(configurations)]


def compute_score(y_validation, predictions):
  balanced_accuracy_score_ = balanced_accuracy_score(y_validation, predictions)
  f1_score_ = f1_score(y_validation, predictions, average='macro')
  accuracy_score_ = accuracy_score(y_validation, predictions)
  precision_score_ = precision_score(y_validation, predictions, average='macro')
  recall_score_ = recall_score(y_validation, predictions, average='macro')

  score_dict = {
        'balanced_accuracy_score': balanced_accuracy_score_,
        'f1_score': f1_score_,
        'accuracy_score': accuracy_score_,
        'precision_score': precision_score_,
        'recall_score': recall_score_
    }

  return score_dict


def extract_values_oblique(data):
    result = []

    for outer_list in data:
        for inner_list in outer_list:
            for item in inner_list:
                node_id = int(item.split(' ')[1])  # Extracting the node ID as an integer

                if 'pivot:' in item:
                    value = int(item.split('pivot: ')[-1])  # Extracting the pivot value as an integer
                    result.append({'node_id': node_id, 'value': value, 'type': 'pivot'})

                elif 'medoid:' in item:
                    value = int(item.split('medoid: ')[-1])  # Extracting the medoid value as an integer
                    result.append({'node_id': node_id, 'value': value, 'type': 'medoid'})

    return result

def extract_values_parallel(data):
    result = []

    for item in data:
        node_id = int(item[0].split(' ')[1])  # Extracting the node ID as an integer

        if 'pivot:' in item[0]:
            value = int(item[0].split('pivot: ')[-1])  # Extracting the pivot value as an integer
            result.append({'node_id': node_id, 'value': value, 'type': 'pivot'})

        elif 'medoid:' in item[0]:
            value = int(item[0].split('medoid: ')[-1])  # Extracting the medoid value as an integer
            result.append({'node_id': node_id, 'value': value, 'type': 'medoid'})

    return result



def pivot_tree_selection(params_list, pairwise_dist_matrix, X_curr, y_curr, random_state):

  #initialize empty lists
  X_pivots_discriminative, y_pivots_discriminative = [], []
  X_pivots_descriptive, y_pivots_descriptive = [], []
  X_pivots_used, y_pivots_used = [], []

  discriminative_idxs = []


  #set parameters
  max_depth_clf = params_list['max_depth']
  allow_oblique_splits_clf = params_list['allow_oblique_splits']
  force_oblique_splits_clf = params_list['force_oblique_splits']
  max_oblique_features_clf = params_list['max_oblique_features']
  pivot_oblique_choice_features_clf = params_list['pivot_oblique_choice_features']
  approximation_clf = params_list['approximation']


  random_state_clf = random_state


  #init pivottree and fit it
  pivot_tree = PivotTree(allow_oblique_splits = allow_oblique_splits_clf, force_oblique_splits = force_oblique_splits_clf, random_state = random_state_clf,
                             max_oblique_features=max_oblique_features_clf,max_depth = max_depth_clf, distance_matrix = pairwise_dist_matrix,
                             pivot_oblique_choice_features = pivot_oblique_choice_features_clf, prune_tree = True, pairwise_metric = metric)


  time_selection = 0.0
  start_time = time.process_time()
  pivot_tree.fit(X_curr, y_curr) #fit the pivot_tree with current configuration
  end_time = time.process_time()

  time_selection = end_time - start_time

  #retrieve nodes
  nodes = [node for node in list(pivot_tree._node_dict.values())] #get nodes

  #get descriptive
  X_descriptive = [node.X_pivot_descriptive for node in nodes if type(node.X_pivot_descriptive) != type(None)]
  y_descriptive  = [node.y_pivot_descriptive for node in nodes if type(node.y_pivot_descriptive) != type(None)]

  #get discriminative
  X_discriminative_pivot_indexes = np.concatenate([node.discriminative_pivot_indexes for node in nodes if type(node.discriminative_pivot_indexes) != type(None)]).astype(int)

  X_discriminative = X_curr[X_discriminative_pivot_indexes]
  y_discriminative = y_curr[X_discriminative_pivot_indexes]

  X_pivots_discriminative += [x for x in X_discriminative]
  y_pivots_discriminative += [x for x in y_discriminative]

  X_pivots_descriptive += sum([x for x in X_descriptive], [])
  y_pivots_descriptive += sum([x for x in y_descriptive],[])

  idxs_pivots_used = [node.pivot_used for node in nodes if type(node.pivot_used) != type(None)]

  if force_oblique_splits_clf:
      idxs_pivots_used = [x['value'] for x in extract_values_oblique(idxs_pivots_used)]
  else:
      idxs_pivots_used = [x['value'] for x in extract_values_parallel(idxs_pivots_used)]

  X_pivots_used += [x for x in X_curr[idxs_pivots_used]]
  y_pivots_used += [x for x in y_curr[idxs_pivots_used]]

  #return fitted pivot_tree and prototypes extracted

  prototypes = {'X_pivots_discriminative': X_pivots_discriminative, 'y_pivots_discriminative' : y_pivots_discriminative,
                'X_pivots_descriptive': X_pivots_descriptive, 'y_pivots_descriptive' : y_pivots_descriptive,
                'X_pivots_used' : X_pivots_used, 'y_pivots_used' : y_pivots_used,
                'X_pivots_discriminative_descriptive' : X_pivots_discriminative + X_pivots_descriptive,
                'y_pivots_discriminative_descriptive' : y_pivots_discriminative + y_pivots_descriptive,

                }

  return pivot_tree, prototypes, time_selection


proto_sets = [['X_pivots_discriminative','y_pivots_discriminative'],
 ['X_pivots_descriptive','y_pivots_descriptive'],
  ['X_pivots_used','y_pivots_used'],['X_pivots_discriminative_descriptive',
 'y_pivots_discriminative_descriptive']]


def evaluate_fitted_tree(X_train, y_train, X_validation, y_validation, pivot_tree, prototypes, configuration):
  #initialize needed classifiers

  results_list = []

  classifiers = {'dt_clf' : DecisionTreeClassifier(max_depth = 4, min_samples_leaf = 3, min_samples_split = 5, random_state = 0),
  'k_5' : KNeighborsClassifier(n_neighbors=5, metric = metric)
  }

  #compute pivot_tree standalone score

  num_leaves = len([True for x in [node.is_leaf for node in list(pivot_tree._node_dict.values())] if x == True])

  time_predict = 0.0
  start_time = time.process_time()
  predictions = pivot_tree.predict(X_validation)
  end_time = time.process_time()
  time_predict = end_time - start_time

  time_fit = time_selection #for pivot_tree, the selection time equals training time
  
  PT_score_dict = compute_score(y_validation, predictions)

  PT_score_dict['configuration'] = 'PT_' + str(configuration)
  PT_score_dict['total_pivots_num'] = len(prototypes['y_pivots_used'])
  PT_score_dict['random_state'] = random_state
  PT_score_dict['num_leaves'] = num_leaves
  PT_score_dict['time_selection'] = time_selection
  PT_score_dict['time_partition'] = time_partition
  PT_score_dict['time_fit'] = time_fit
  PT_score_dict['time_predict'] = time_predict
  PT_score_dict['time_pairwise_training'] = 0.0
  PT_score_dict['time_pairwise_validation'] = 0.0
  PT_score_dict['data_set_name'] = data_set_name

  results_list += [PT_score_dict]

  #KNN PERFORMANCE ON ORIGINAL SPACE

  for neigh in [1,3,5]:
    for tup in proto_sets:
      X_pivots_name = tup[0]
      y_pivots_name = tup[1]

      if len(prototypes[y_pivots_name]) < neigh:
        continue

      num_leaves = 0.0

      knn = KNeighborsClassifier(n_neighbors=neigh, metric = metric)

      time_fit = 0.0
      start_time = time.process_time()
      knn.fit(prototypes[X_pivots_name], prototypes[y_pivots_name])
      end_time = time.process_time()
      time_fit = end_time - start_time

      time_predict = 0.0
      start_time = time.process_time()
      predictions = knn.predict(X_validation)
      end_time = time.process_time()
      time_predict = end_time - start_time

      KNN_score_dict = compute_score(y_validation, predictions)

      KNN_score_dict['configuration'] = 'PT_KNN' + str(neigh) + '_original_' + y_pivots_name + '_' + str(configuration)
      KNN_score_dict['total_pivots_num'] = len(prototypes[y_pivots_name])
      KNN_score_dict['random_state'] = random_state
      KNN_score_dict['num_leaves'] = num_leaves
      KNN_score_dict['time_selection'] = time_selection
      KNN_score_dict['time_partition'] = time_partition
      KNN_score_dict['time_fit'] = time_fit
      KNN_score_dict['time_predict'] = time_predict
      KNN_score_dict['time_pairwise_training'] = 0.0
      KNN_score_dict['time_pairwise_validation'] = 0.0
      KNN_score_dict['data_set_name'] = data_set_name


      results_list += [KNN_score_dict]

   #KNN PERFORMANCE ON SPACE_TRANSFORMED

  for neigh in [1,3,5]:
    for tup in proto_sets:
      X_pivots_name = tup[0]
      y_pivots_name = tup[1]

      if len(prototypes[y_pivots_name]) < neigh:
        continue

      X_train_reduced, time_pairwise_training = pariwise_computation(X_train, prototypes[X_pivots_name], metric) #reduce training set
      X_validation_reduced, time_pairwise_validation = pariwise_computation(X_validation, prototypes[X_pivots_name], metric) #reduce validation set

      #if i extract less prottoype then config for KNN, i skip it
     
          
      num_leaves = 0.0

      knn = KNeighborsClassifier(n_neighbors=neigh, metric = metric)

      time_fit = 0.0
      start_time = time.process_time()
      knn.fit(X_train_reduced, y_train)
      end_time = time.process_time()
      time_fit = end_time - start_time

      time_predict = 0.0
      start_time = time.process_time()
      predictions = knn.predict(X_validation_reduced)
      end_time = time.process_time()
      time_predict = end_time - start_time

      KNN_score_dict = compute_score(y_validation, predictions)

      KNN_score_dict['configuration'] = 'PT_KNN' + str(neigh) + '_mapped_' + y_pivots_name + '_' + str(configuration)
      KNN_score_dict['total_pivots_num'] = len(prototypes[y_pivots_name])
      KNN_score_dict['random_state'] = random_state
      KNN_score_dict['num_leaves'] = num_leaves
      KNN_score_dict['time_selection'] = time_selection
      KNN_score_dict['time_partition'] = time_partition
      KNN_score_dict['time_fit'] = time_fit
      KNN_score_dict['time_predict'] = time_predict
      KNN_score_dict['time_pairwise_training'] = time_pairwise_training
      KNN_score_dict['time_pairwise_validation'] = time_pairwise_validation
      KNN_score_dict['data_set_name'] = data_set_name


      results_list += [KNN_score_dict]


   #DT PERFORMANCE ON SPACE TRANSFORMED

  for depth in [4]:
      for tup in proto_sets:
        X_pivots_name = tup[0]
        y_pivots_name = tup[1]


        if len(prototypes[y_pivots_name]) < 1:
            continue

        X_train_reduced, time_pairwise_training = pariwise_computation(X_train, prototypes[X_pivots_name], metric) #reduce training set
        X_validation_reduced, time_pairwise_validation = pariwise_computation(X_validation, prototypes[X_pivots_name], metric) #reduce validation set

        dt = DecisionTreeClassifier(max_depth = depth, min_samples_leaf = 3, min_samples_split = 5, random_state = 0)

        time_fit = 0.0
        start_time = time.process_time()
        dt.fit(X_train_reduced, y_train)
        end_time = time.process_time()
        time_fit = end_time - start_time

        time_predict = 0.0
        start_time = time.process_time()
        predictions = dt.predict(X_validation_reduced)
        end_time = time.process_time()
        time_predict = end_time - start_time

        num_leaves = dt.get_n_leaves()

        DT_score_dict = compute_score(y_validation, predictions)

        DT_score_dict['configuration'] = 'PT_DT' + str(depth) + '_mapped_' + y_pivots_name + '_' + str(configuration)
        DT_score_dict['total_pivots_num'] = len(prototypes[y_pivots_name])
        DT_score_dict['random_state'] = random_state
        DT_score_dict['num_leaves'] = num_leaves
        DT_score_dict['time_selection'] = time_selection
        DT_score_dict['time_partition'] = time_partition
        DT_score_dict['time_fit'] = time_fit
        DT_score_dict['time_predict'] = time_predict
        DT_score_dict['time_pairwise_training'] = time_pairwise_training
        DT_score_dict['time_pairwise_validation'] = time_pairwise_validation
        DT_score_dict['data_set_name'] = data_set_name


        results_list += [DT_score_dict]
  
   
  return results_list



#for each random state, fit a pivot tree on the validation set and select the prototypes

result_list = []
for random_state in range(0,5):
  #for each configuration we fit a pivot tree with that configuration and random state
  for params_list in possible_configurations:

    #obtain fitted tree with prototypes

    pivot_tree, prototypes, time_selection = pivot_tree_selection(params_list, pairwise_dist_matrix = X_train_inner_pairwise_matrix,
                                              X_curr = X_train_inner, y_curr = y_train_inner, random_state = random_state)
    

    #obtain results for current configuration and random state
    result_list = result_list + evaluate_fitted_tree(X_train_inner, y_train_inner,
                                    X_validation,y_validation,
                                    pivot_tree, prototypes, params_list)
    



unique_configs = set([x['configuration'] for x in result_list])

#for each unique configuration, compute the mean across all random states

def compute_average_and_std(data_list):
    # Define the keys to consider for metrics
    keys_to_consider = [
        'accuracy_score',
        'balanced_accuracy_score',
        'f1_score',
        'precision_score',
        'recall_score',
        'time_partition',
        'time_selection',
        'time_pairwise_training',
        'time_pairwise_validation',
        'time_fit',
        'time_predict',
        'total_pivots_num',  # Include total_pivots_num in the keys
        'num_leaves',
    ]

    # Define the keys to keep constant
    keys_constant = [
        'configuration',  
        'data_set_name',
        
    ]

    # Initialize a dictionary to store the computed values
    result_dict = {}

    # Include constant keys in the result
    for key in keys_constant:
        result_dict[key] = data_list[0][key]  # Assuming the constant values are the same for all entries

    # Extract the relevant metrics into separate lists for each key
    key_lists = {key: [entry[key] for entry in data_list] for key in keys_to_consider}

    # Compute the average and standard deviation for each key
    for key, values in key_lists.items():
        average_key = np.mean(values)
        std_key = np.std(values)
        result_dict[f'average_{key}'] = average_key
        result_dict[f'std_{key}'] = std_key

    return result_dict


def matching_entries(data, configuration_classifier_name):
    configuration = configuration_classifier_name[0]
    classifier_name = configuration_classifier_name[1]
    matching_entries = [entry for entry in data if entry['configuration'] == configuration and entry['classifier_name'] == classifier_name]
    if not matching_entries:
        print(f"No matching entries found for configuration: {configuration} and classifier: {classifier_name}")
        return None
    return matching_entries



dictionaries_result = []

for config in unique_configs:
  current_results = [x for x in result_list if x['configuration'] == config]
  #print(len(current_results))
  #print(current_results)

  dictionaries_result.append(compute_average_and_std(current_results))



print(possible_configurations[0])

avg_dict = pd.DataFrame(dictionaries_result)
print(avg_dict.shape)
print(avg_dict.iloc[0])
print(data_set_name)
print(mode_of_evaluation)
avg_dict.to_csv(data_set_name + '_PT_'+mode_of_evaluation+ '.csv', index = False)














