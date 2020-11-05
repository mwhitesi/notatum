import numpy as np
import pandas as pd

from Bio import AlignIO
from timeit import timeit

class OrthologTokenizer:
    """
        Encode protein alignment blocks into tokens

    """

    def __init__(self):
        pass

    def encode_ortholog(self, gn, aln, max_major_allele_freq=0.99, min_major_allele_freq=0.50):

        # Load alignment strings into individual char columns
        adf = pd.DataFrame(aln)
        adf = adf.iloc[:,0].str.split('', expand=True)
        adf = adf.iloc[:,1:-1]
        
        # Calculate column-wise frequencies and assign allele labels
        freqs = pd.DataFrame([ adf[col].replace(to_replace=adf[col].value_counts(normalize=True).sort_index().rank(method="first", ascending=False).to_dict()) for col in adf ]).transpose()

        # Identify columns to filter
        mask = np.array([ True if max(adf[col].value_counts(normalize=True)) >= min_major_allele_freq and max(adf[col].value_counts(normalize=True)) <= max_major_allele_freq else False for col in adf ])
        freqs = freqs.loc[:,mask]

        # Identify runs of identical conservation blocks
        col_patterns = freqs.apply(lambda x: hash(tuple(x)), axis = 0).to_numpy()
        _, p, _ = self.rle(col_patterns)

        # Output block variant ids
        # Save 0 for missing -- if genome is missing gene it gets 0 for all blocks (i don't want to use all gaps in pattern deconvolution step)
        i = 1
        bk_vars = freqs.iloc[:,p]
        inc = bk_vars.max(axis=0)
        inc = inc.cumsum() + i
        inc = np.append(np.array([0]), inc.to_numpy())[0:-1]
        bk_vars = bk_vars + inc
        bk_vars = bk_vars.astype('uint32')

        return(bk_vars)


    def rle(self, ia):
        """
            Run length encoding

            returns: tuple (run_lengths, start_positions, values)

        """

        n = len(ia)
        y = np.array(ia[1:] != ia[:-1])
        i = np.append(np.where(y), n - 1)
        z = np.diff(np.append(-1, i))
        p = np.cumsum(np.append(0, z))[:-1]
        return(z, p, ia[i])


if __name__ == "__main__":

    ai = AlignIO.parse('data/tmp/test.aln', 'clustal')

    msa = next(ai)
    gn = [record.id for record in msa]
    aln = [str(record.seq) for record in msa]

    ot = OrthologTokenizer()
    vocab, ctx_len, vocab_len = ot.encode_ortholog(gn, aln)

    
