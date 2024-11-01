import pandas as pd


def find_discrete(df):
    df = pd.DataFrame(df)
    int_cols = [col for col in df if df[col].apply(lambda x: x.is_integer() if isinstance(x, float) else
                isinstance(x, int)).all()]
    return int_cols

