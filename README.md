![image](public/CHOFormer_logo.png)
[![LICENSE](https://img.shields.io/badge/license-MIT-brightgreen)](https://github.com/Lattice-Automation/icor-codon-optimization/blob/master/LICENSE)

# CHOFormer: Optimizing Protein Expression in CHO Cells
[![Button Component](https://readme-components.vercel.app/api?component=button&text=Use This Tool: choformer.com)](https://choformer.com)

<!-- ![image](public/flowchart.png) -->

- ### <h3> <a href="#about">About</a></h3>
- ### <h3> <a href="#usetool">How To Use This Tool</a> </h3>
- ### <h3> <a href="#benchmarking">Benchmarking</a> </h3>
- ### <h3> <a href="#training">Training</a> </h3>
- ### <h3> <a href="#license">License</a> </h3>

## <h2 id="about">About</a> </h2>

The genetic code is degenerate; there are 61 sense codons encoding for only 20 standard amino acids. While synonymous codons encode the same amino acid, their selection can drastically influence the **speed and accuracy** of protein production. CHOFormer is a cutting-edge **Transformer decoder model** developed to optimize codon sequences for **improving protein expression** in Chinese Hamster Ovary (CHO) cells. As CHO cells are used in the production of nearly 70% of recombinant pharmaceuticals, including monoclonal antibodies and other therapeutic proteins, optimizing protein yield in these cells is a critical step in drug development. However, low protein yields in CHO cells present a significant challenge, slowing down the drug manufacturing process. CHOFormer addresses these challenges by leveraging a transformer decoder model to optimize codon selection based on the relationship between protein expression and codon usage patterns. This results in significantly improved protein yields, and the optimization process, which traditionally takes months in laboratory settings, is reduced to **mere minutes** with CHOFormer.

<!-- CHOFormer is a state-of-the-art **transformer decoder model** designed to optimize codon sequences for enhanced protein expression in Chinese Hamster Ovary (CHO) cells. Today, nearly 70% of recombinant pharmaceuticals are manufactured using the CHO genome in their research and development. This tool addresses the challenge of low recombinant protein yields in CHO cells, critical for drug manufacturing, particularly in the development of monoclonal antibodies and other therapeutic proteins. -->

<!-- Codon optimization, currently time-consuming in laboratory environments, is significantly expedited by using CHOFormer, potentially shortening the optimization timeline from **months to minutes**. -->
![public/architecture.png](public\architecture.png)
## <h2 id="usetool">How To Use This Tool</a> </h2>
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=flat&logo=flask&logoColor=white) ![Next JS](https://img.shields.io/badge/Next-black?style=flat&logo=next.js&logoColor=white) ![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54) ![AWS](https://img.shields.io/badge/AWS-%23FF9900.svg?style=flat&logo=amazon-aws&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)

Our [website](https://choformer.com) allows for the easy usage of the tool.
For **CHOFormer**, simply input the protein sequence and the optimized DNA sequence will be outputted—it can be downloaded as a FASTA or copied directly.
For **CHOExp**, upload or paste the DNA sequence; the output will be the protein expression (normalized between 0 and 1).

## Technical Overview
1. Accessed 97000 CHO gene sequences from [NCBI](https://ftp.ncbi.nlm.nih.gov/genomes/genbank/vertebrate_mammalian/Cricetulus_griseus/all_assembly_versions/). Filter for only protein-coding genes.
2. Filter sequences to be between 300 and 8000 base pairs (86632 sequences).
3. Run `cd-hit-est` to cluster the sequences at 8 words and 0.9 nucleotide similarity (47713 sequences).
3. Translate to amino acids, removing unnatural amino acids (47713 sequences)
4. Use ESM-2-150M to extract embeddings from protein sequences.
5. Protein and DNA mappings are split into training, validation, and test splits (80-10-10)
6. The model decodes the DNA sequence based on the input embeddings from ESM2. It has a dimension of 128 with 2 layers and 4 attention heads.
7. DNA Vocabulary Mapping: The decoder output is mapped to the DNA codon vocabulary.
8. The final optimized DNA sequence is generated for high expression in CHO cells.

# CHO Expression Predictor (CHOExp)
CHOExp is a transformer model designed to predict the expression levels of optimized DNA sequences in CHO cells based on RNA-Seq data, leveraging a correlation between RNA and protein expression. The model has 3 layers with a model dimension of 256 and 4 attention heads.

## Technical Overview
1. Accessed 26795 genes with RNA expression values.
2. Removed genes with zero expression and only included the top 66% and those within three standard deviations — 13253 genes.
3. Split expression data into training, validation, and test splits (80-10-10)
4. Predict expression for proteins in the training set using an encoder-only transformer model (dimension of 384 with 8 layers and 4 attention heads)
5. Evaluate the difference between ground truth and predicted expression values to improve the model during the training process
6. CHOExp is then used to select high-expression CHO genes to use during CHOFormer's training process
