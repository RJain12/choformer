# Project Story
The inspiration for CHOFormer came from a pressing challenge in the biopharmaceutical industry: low recombinant protein yields in Chinese Hamster Ovary (CHO) cells. CHO cells are the backbone of biologics production, responsible for nearly 70% of recombinant pharmaceuticals, including monoclonal antibodies and other therapeutic proteins. However, optimizing protein expression in these cells can be a time-consuming process, often taking months to years in laboratory settings. The complexity of codon optimization, where synonymous codons can dramatically affect protein expression, was an exciting problem we knew machine learning could solve. We wanted to build a tool that not only optimized codon sequences but could do so in minutes, speeding up the entire drug development pipeline.

## What We Learned
Throughout the project, we delved deep into the world of protein expression and learned how codon usage bias—the preference for certain codons over others—can significantly affect protein yield. Despite these codons encoding the same amino acids, their selection impacts translation speed and accuracy, influencing protein folding and stability. Moreover, we even made a supplementary transformer model (CHOExp) after we realised existing models to predict protein expression e.g. Evo were solely for prokaryotes.

The use of transformer decoders allowed us to explore these relationships in new ways, creating a solution that learns directly from the data instead of relying on fixed heuristics. 

## How We Built It
Building CHOFormer involved several steps. First, we gathered 97,000 CHO gene sequences from the NCBI database, filtering them for protein-coding genes and narrowing them down to 86,632 sequences. After clustering these sequences using `cd-hit-est` with a 0.9 nucleotide similarity threshold, we translated the sequences to their corresponding amino acids, removing any unnatural sequences, resulting in 47,713 protein sequences. From there, we used the `ESM-2-650M model` to extract rich embeddings from the protein sequences.

For the DNA sequence generation, we tokenized the DNA sequences using DNABert, ensuring that positional information was embedded to maintain the sequence order. The Transformer decoder model took these embeddings and generated optimized DNA sequences, mapping them to the codon vocabulary for final output. The model was split into 80-10-10 for training, validation, and testing, using 2 layers, 4 attention heads, and an embedding dimension of 128.

Parallelly, we developed CHOExp, a model designed to predict expression levels of the generated DNA sequences. Using RNA-Seq data from 26,795 genes, we refined this dataset to 13,253 genes after filtering for non-zero expression values and removing outliers. CHOExp uses a 3-layer Transformer model with a dimension of 256 and 4 attention heads, capable of predicting RNA expression based on gene sequences.

## Challenges We Faced
One of the most significant challenges we faced was data preprocessing. The gene sequence data from public databases came in various formats and contained sequences that were incomplete, too short, or too long. Ensuring that the sequences we used were consistent and valid required extensive cleaning and filtering. Additionally, the clustering step using cd-hit-est was highly computationally expensive, and so we had to iteratively change our input (words and nucleotide similarity) to minimize compute time.

On the model side, training a Transformer decoder model to handle both protein embeddings and DNA sequence generation required careful architectural design. Handling long DNA sequences without overwhelming the model was a particular concern, as was ensuring that the model could generalize well to unseen sequences without overfitting to the training data.













