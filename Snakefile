import source.io.dataloader as dl

DRUGS = ['CTZ', 'CTX', 'AMP', 'AMX', 'AMC', 'TZP', 'CXM', 'CET', 'GEN', 'TBM',
         'TMP', 'CIP']

rule transform:
    input:
        "data/raw/ecoli/Metadata.csv",
        "data/raw/ecoli/AccessoryGene.csv"
    output:
        expand("data/interim/ecoli/drugs/{drug}.pt", drug=DRUGS)
    run:
        dl.transform(input, output)


rule align:
    input:
        "data/raw/ecoli/pan_genome_sequences.zip"
    output:
        dynamic("data/interim/gene/{GENE}.aln")
    run:
        # TODO, 
        pass

rule tokenize:
    input:
        dynamic("data/interim/gene/{GENE}.aln")
    output:
        dynamic("data/interim/gene/{GENE}.blocks")
    run:
        # TODO
        pass



