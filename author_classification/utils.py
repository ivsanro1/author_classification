import pandas as pd

def expand_col_dicts(df, colname_dict):
    return pd.concat([
        df.reset_index(),
        pd.json_normalize(df[colname_dict]).reset_index(drop=True),
    ], axis=1).set_index('index').drop(colname_dict, axis=1)