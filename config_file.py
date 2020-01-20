"""Configuration File

File containing a set of configuration variables used by
the files in this project.

"""
import os

#
#   Paths configuration.
#

# Path to model
model_path = '/model_selector2.sav'

# Path and configurations for the logger function.
logg_it_output_path = os.path.abspath(os.getcwd())
logg_it_output_filename = '/logger.txt'
log_printing = False

# Path and name of the generated trained results file.
result_saver_output_path = os.path.abspath(os.getcwd())
result_saver_output_filename = '/trained_results.txt'


#
#   Timeout configurations.
#

# Maximum number of seconds elapsed before an algorithm is stopped.
algorithm_timeout = 240

# Maximum number of seconds elapsed before the automl function is stopped.
automl_timeout = 1140

# Value returned by a function if its stopped.
timeout_return = -1


#
#   Algorithms configurations.
#

# Number of default folds, optional argumnet in function allows to modify it.
k_folds = 4

# List of the training algorithms.
training_mdls = ['automatedKNN', 'automatedLogReg', 'automatedBerNB',
                 'automatedGaussNB', 'automatedPassiveAgr',
                 'automatedRidgeReg', 'automatedSGDReg', 'automatedSVM',
                 'automatedDecisionTree', 'automatedRandomForest',
                 'automatedBagging', 'automatedHistGB']

# Predefined imput variable for the random search.
search_cv = 5

# Number of cores to use (-1 for all)
core_count = -1


#
#   automl Configurations.
#

# Option to return the AUC or the algorithm name and the AUC.
automl_explicit = False

# Default algorithm to be executed first
default_algorithm = ['automatedKNN']

# Primitives to be applied to add new features.
trns_primitives = [['add_numeric'], ['multiply_numeric'],
                   ['divide_numeric'], ['add_numeric', 'multiply_numeric'],
                   ['add_numeric', 'divide_numeric'],
                   ['multiply_numeric', 'divide_numeric'],
                   ['add_numeric', 'multiply_numeric', 'divide_numeric']]
