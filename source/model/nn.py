
from torch import nn

def build_model(config):

    nlayers = config['nlayers']

    # global params
    dropout = config.get('all_dropout', 0)
    batchnorm = config.get('all_batchnorm', 0)
    activation_func = config.get('activation_func', 'relu')

    # build layers
    n_prev = config['0_nodes']
    n = config['1_nodes']
    layers = [nn.Linear(n_prev, n)]

    for l in range(2:nlayers):
        n_prev = n
        n = config['{}_nodes'.format(l)]
        l_batchnorm = batchnorm or config.get('{}_batchnorm'.format(l), 0)
        l_dropout = dropout or or config.get('{}_dropout'.format(l), 0)
        new_layers = add_layer(n_prev, n, activation_func, l_dropout, l_batchnorm)
        layers.extend(new_layer)

    return(nn.Sequential(layers))


def add_layer(n_in, n_out, activation, dropout, batchnorm):

    if activation == 'relu':
        layers = [nn.ReLU()]
    else:
        raise "Unknown activation function"

    if dropout and dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    if batchnorm:
        raise "Batch normalization not implemented"

    layers.append(nn.Linear(n_in, n_out))

    return layers













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
