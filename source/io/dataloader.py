from mxnet import gluon
import pandas as pd

class GeneDataset(gene.data.ArrayDataset):

    def __init__(self, gene_filepath, meta_filepath):

        # Load inputs
        X = 0
        y = 0
        # Initialize ArrayDataset object
        super().__init__(X, y)



class Transformer():

    def __init__(self, gene_filepath, meta_filepath):
