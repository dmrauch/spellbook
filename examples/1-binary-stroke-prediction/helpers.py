import numpy as np
import pandas as pd
import tensorflow as tf

import spellbook as sb


def load_data():

    data = pd.read_csv('healthcare-dataset-stroke-data.csv')

    # preliminary data inspection and cleaning
    print(data.dtypes)  # datatype of each column
    print(data.head)    # table with first few lines of the data
    print(data.count()) # count valid datapoints - something going on with 'bmi'
    print(data['bmi'])  # print column - 'bmi' contains floats and 'NaN'

    # drop rows containing NaN values, inplace
    data.dropna(inplace=True)

    # after data exploration, remove column 'id'
    data.drop(columns=['id'], inplace=True)

    # clean up variable names
    rename_dict = {'Residence_type': 'residence_type'}
    data.rename(columns=rename_dict, inplace=True)

    # clean up datatypes
    replace_dict = {
        'ever_married': {'No': 'no', 'Yes': 'yes'},
        'gender': {
            'Female': 'female',
            'Male': 'male',
            'Other': 'other'
        },
        'heart_disease': {0: 'no', 1: 'yes'},
        'hypertension': {0: 'no', 1: 'yes'},
        'residence_type': {
            'Urban': 'urban',
            'Rural': 'rural'
        },
        'smoking_status': {
            'Unknown': 'unknown',
            'never smoked': 'never',
            'formerly smoked': 'formerly'
        },
        'work_type': {
            'Govt_job': 'govt',
            'Never_worked': 'never',
            'Private': 'private',
            'Self-employed': 'self'
        },
        'stroke': {0: 'no', 1: 'yes'}
    }
    data.replace(replace_dict, inplace=True)

    # create lists of variable names
    vars = list(data)
    target = 'stroke'
    features = vars.copy()   # make a copy to protect the original list
    features.remove(target)  # from being modified by the 'remove' statement
    print(target)
    print(features)
    print(vars)

    # print what kinds the variables are (cat/ord/cont)
    sb.plotutils.print_data_kinds(data)

    return((data, vars, target, features))
