"""Common Functions

Functions used to save valuable information to files.

    * log_it - logger funtion to track the execution process.
    * save_results - saves the trained results
"""

import datetime
from config_file import logg_it_output_path, logg_it_output_filename, \
    result_saver_output_path, result_saver_output_filename, log_printing


def log_it(instance, message, explicit=log_printing):
    """Logs and may print the given information.

    Parameters
    ----------
    instance : sring
        String to save in the first position. Should follow a constant
        methodology to ease debugging. For example:
            'Method: ' + function_name.
            'Error: ' + error information.
    message : sring
        String to save in the second position. Message returned or
        to be printed related to instance. For example:
            'Starting execution.'
    explicit : boolean, optional
        Used to define if the function should also print the information
        on the console.

    OutPuts
    -------
        Appends each register to the file specified in the config_file
        by the variables:
            logg_it_output_path, logg_it_output_filename
    """
    st = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if explicit:
        print(st+"; "+str(instance)+"; Logged: "+message)
    f = open(logg_it_output_path+logg_it_output_filename, "a")
    f.write(st+"; "+str(instance)+"; Logged: "+message+"\n")
    f.close()


def save_results(file_name, row_num, col_num,
                 num_classes, algorithm_name, AUC, time):
    """Logs the training results.

    Parameters
    ----------
    file_name : sring
        Filename of the analyzed DataSet.
    row_num : int
        Number of rows of the treated DataSet.
    col_num : int
        Number of columns of the treated DataSet.
    num_classes : int
        Number of classification classes of the DataSet.
    algorithm_name : string
        Name of the applied algorithm.
    AUC : float
        Resulting AUC mean accoplished by the algorithm.
    time : float
        Resulting mean execution time.

    OutPuts
    -------
        Appends each register to the file specified in the config_file
        by the variables:
            result_saver_output_path, result_saver_output_filename
    """
    time = datetime.timedelta(seconds=time)
    f = open(result_saver_output_path+result_saver_output_filename, "a")
    f.write(""+str(file_name)+";"+str(row_num)+";"+str(col_num)+";" +
            str(num_classes)+";"+str(algorithm_name)+";"+str(AUC)+";" +
            str(time)+"\n")
    f.close()
