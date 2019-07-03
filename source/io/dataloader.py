import pandas as pd
import numpy as np
import re
import torch

# class GeneDataset(gluon.data.ArrayDataset):
#
#     def __init__(self, gene_filepath, meta_filepath):
#
#         # Load inputs
#         X = 0
#         y = 0
#         # Initialize ArrayDataset object
#         super().__init__(X, y)

def transform(input, output):
    """Snakemake function

        Split and transform input data

    """

    genesdf = pd.read_csv(input[1], index_col=0, header=0)
    metadf = pd.read_csv(input[0])

    all_isolates = metadf["Isolate"].to_numpy('U')

    encoding = {
      'S': 0,
      'I': 0.5,
      'R': 1
    }

    pattern = re.compile("(\w{3}).pt$")

    for f in output:
        m = pattern.match(f, len(f)-6)
        d = m.group(1)
        # print(d)
        y = metadf[d]
        omit = pd.isnull(y)
        isolates = all_isolates[~omit]
        y = y.loc[~omit]
        X = genesdf.loc[isolates].to_numpy()

        ylabels = np.array([ encoding[v] for v in y ])

        # print(ylabels.shape)
        # print(X.shape)
        # print(isolates.shape)
        # print(isolates[0])
        # print(isolates.dtype)

        y_tensor = torch.from_numpy(ylabels)
        X_tensor = torch.from_numpy(X)

        torch.save({'y': y_tensor, 'X': X_tensor, 'isolates': isolates}, f)


if __name__ == "__main__":

    input = [
        "data/raw/ecoli/Metadata.csv",
        "data/raw/ecoli/AccessoryGene.csv"
    ]
    output = [
        "data/interim/ecoli/drugs/CTX.pt",
        "data/interim/ecoli/drugs/AMP.pt",
        "data/interim/ecoli/drugs/AMX.pt"
    ]
    transform(input, output)
