
import source.io.Transformer

DRUGS=['']

rule raw_data_transform:
    input:
        "data/raw/Metadata.csv",
        "data/raw/AccessoryGenes.csv"
    output:
        "data/interim/ecoli/gene.csv",
        expand("data/interim/ecoli/drugs/{drug}.csv", drug=DRUGS)
    run:
        transformer = Transformer()
        transformer.transform(input[0], input[1], output)
