import source.io.dataloader as dl

DRUGS=['CTZ', 'CTX', 'AMP', 'AMX', 'AMC', 'TZP', 'CXM', 'CET', 'GEN', 'TBM', 'TMP', 'CIP']

rule transform:
    input:
        "data/raw/ecoli/Metadata.csv",
        "data/raw/ecoli/AccessoryGene.csv"
    output:
        expand("data/interim/ecoli/drugs/{drug}.npz", drug=DRUGS)
    run:
        dl.transform(input, output)
