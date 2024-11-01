from pyclustkit.metalearning import MFExtractor
import pandas as pd
import numpy as np


if __name__ == "__main__":
    df = pd.read_csv(r"D:\PyClust-Eval\tests\iris.csv")
    mfe = MFExtractor(data=np.array(df))
    mfe.calculate_mf()

