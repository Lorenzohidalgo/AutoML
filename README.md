# Automated Machine Learning
Python automated machine learning script used to predict categorical data Sets.

### Installation
To be able to run our script you will need to install some basic python library requirements.
This can be done by running the .bat file included in the repository or by running the following command:

```sh
$ pip install -r requirements.txt
```

After installing all dependencies make sure to open the config_file.py and update the variable named model_path. This path shoul reference the model_selector.sav file.

Following a simple example on how to run the automl function:

```sh
>> import pandas as pd
>> from automl import automl
>> df = pd.read_csv('my_csv.csv', sep=';')
>> AUC = automl(df) # use explicit=True to get the name of the used algorithm.
```

For further information please refer to the PDF report.

License
----

GNU General Public License v3.0
