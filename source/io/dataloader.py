import pandas as pd
import numpy as np
import os
import re
import torch
import torch.utils.data as data_utils
import zipfile
import zlib


class IndexTensorDataset:
    """

        Identical to torch.utils.data.Dataset.TensorDataset, but __getitem__
        also returns indices as last value in tuple

    """

    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        t = [tensor[index] for tensor in self.tensors]
        t.append(index)
        return(tuple(t))

    def __len__(self):
        return self.tensors[0].size(0)


class GeneDataset:
    """

        Container object that provides access to the PyTorch Dataset and
        Dataloader objects needed for one experiment

    """

    def __init__(self, data_file, batch_size, test_split, shuffle_dataset,
                 random_seed, validation_split=0):

        # Load tensor data
        data = torch.load(data_file)
        dataset = IndexTensorDataset(data['X'], data['y'])

        # Test / train split
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(test_split * dataset_size))
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, test_indices = indices[split:], indices[:split]

        # Initialize Dataloaders
        train_sampler = data_utils.SubsetRandomSampler(train_indices)
        test_sampler = data_utils.SubsetRandomSampler(test_indices)

        self.train_loader = data_utils.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  sampler=train_sampler)
        self.test_loader = data_utils.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 sampler=test_sampler)
        self.isolates = data['isolates']



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


def align(zipf, transl=True):
    """
        Iterate through pangenome clusters, optionally translate, and align

    """
    with zipfile.ZipFile(zipf, "r") as zh:
        i = 0
        for z in zh.infolist():
            if not z.is_dir():
                gz = zh.read(z.filename)
                fn = zlib.decompress(gz, 15 + 16)
                
            if i > 3:
                break
            i+=1


if __name__ == "__main__":

    align("data/raw/ecoli/pan_genome_sequences.zip")
