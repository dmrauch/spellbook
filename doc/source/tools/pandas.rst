******
pandas
******


Best of *pandas*
================

- Chain an arbitrary number of selection conditions - kudos to https://stackoverflow.com/a/64945576

  .. code:: python

     import pandas as pd
     df = pd.DataFrame({
         'var1': [1, 2, 3, 4, 1, 2, 3, 4],
         'var2': ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b'],
         'var3': [8, 7, 6, 5, 4, 3, 2, 1]
     })

     # specify an arbitrary number of conditions
     variables = ['var1', 'var2']
     values    = [1, 'a']

     # identify the (row) indices for which the conditions are fulfilled
     locator = pd.concat(
         [(df[col] == val) for col, val in zip(variables, values)], axis='columns'
     ).all(axis='columns')

     print(df.loc[locator])