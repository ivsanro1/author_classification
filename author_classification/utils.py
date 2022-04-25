import pandas as pd

def expand_col_dicts(df:pd.DataFrame, colname_dict:str) -> pd.DataFrame:
    '''Expands a column containing dictionaries in the given dataframe, adding each of the keys as a new column.'''
    return pd.concat([
        df.reset_index(),
        pd.json_normalize(df[colname_dict]).reset_index(drop=True),
    ], axis=1).set_index('index').drop(colname_dict, axis=1)
