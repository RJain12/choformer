# CHOFormer
### Optimizing Protein Expression in CHO Cells
![image](public/flowchart.png)
CHOFormer is a state-of-the-art **transformer decoder model** designed to optimize codon sequences for enhanced protein expression in Chinese Hamster Ovary (CHO) cells. Today, nearly 70% of recombinant pharmaceuticals are manufactured using the CHO genome in their research and development. This tool addresses the challenge of low recombinant protein yields in CHO cells, critical for drug manufacturing, particularly in the development of monoclonal antibodies and other therapeutic proteins.

Codon optimization, currently time-consuming in laboratory environments, is significantly expedited by using CHOFormer, potentially shortening the optimization timeline from **months to minutes**.

## Technical Overview
1. Accessed 97000 CHO gene sequences from [NCBI](https://ftp.ncbi.nlm.nih.gov/genomes/genbank/vertebrate_mammalian/Cricetulus_griseus/all_assembly_versions/). Filter for only protein-coding genes.
2. Filter sequences to be between 300 and 8000 base pairs (86632 sequences).
3. Run `cd-hit-est` to cluster the sequences at 8 words and 0.9 nucleotide similarity (47713 sequences).
3. Translate to amino acids, removing unnatural amino acids (47713 sequences)
4. Use ESM-2-650M to extract embeddings from protein sequences.
5. DNA sequence tokens are embedded (DNABert), and positional information is added. Split into 80-10-10.
6. The model decodes the DNA sequence based on the input embeddings. It has a dimension of 128 with 2 layers and 4 attention heads.
7. DNA Vocabulary Mapping: The decoder output is mapped to the DNA codon vocabulary.
8. The final optimized DNA sequence is provided for high expression in CHO cells.

# CHO Expression Predictor (CHOExp)
CHOExp is a transformer model designed to predict the expression levels of optimized DNA sequences in CHO cells based on RNA-Seq data, leveraging a correlation between RNA and protein expression. The model has 3 layers with a model dimension of 256 and 4 attention heads.

## Technical Overview
1. Accessed 26795 genes with RNA expression values.
2. Removed genes with zero expression and only included the top 66% and those within three standard deviations — 13253 genes.
