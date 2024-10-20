# Project Story
The inspiration for CHOFormer came from a pressing challenge in the biopharmaceutical industry: low recombinant protein yields in Chinese Hamster Ovary (CHO) cells. The genetic code is degenerate; there are 61 sense codons encoding for only 20 standard amino acids. While synonymous codons encode the same amino acid, their selection can drastically influence the **speed and accuracy** of protein production. Chinese Hamster Ovary (CHO) cells are responsible for producing nearly 70% of recombinant pharmaceuticals, such as monoclonal antibodies and therapeutic proteins. However, low protein yields in these cells pose a major challenge, often delaying the drug manufacturing process. To address these challenges, we present CHOFormer, a cutting-edge generative model that produces optimized codon sequences for **improved protein expression in CHO cells**. Specifically, we leverage the **Transformer decoder-only model** to optimize codon selection solely based on information-rich protein sequence embeddings. With CHOFormer, we observe a mean Codon Adaptation Index (CAI) of 0.847, indicating that CHOFormer-generated codon sequences are highly adapted for efficient translation in CHO cells. Overall, CHOFormer is a computational pipeline that revolutionizes the codon optimization process, reducing what traditionally takes months in a laboratory setting to **mere minutes**.
The complexity of codon optimization was an exciting problem we knew machine learning could solve. We wanted to build a tool that not only optimized codon sequences but could do so in minutes, speeding up the entire drug development pipeline. Finally, we wanted to release it for free and did so at https://choformer.com

## What We Learned
Throughout the project, we delved deep into the world of protein expression and learned how codon usage bias—the preference for certain codons over others—can significantly affect protein yield. Despite these codons encoding the same amino acids, their selection impacts translation speed and accuracy, influencing protein folding and stability. Moreover, we even made a supplementary transformer model (CHOExp) after we realised existing models to predict protein expression e.g. Evo were solely for prokaryotes.

The use of transformer decoders allowed us to explore these relationships in new ways, creating a solution that learns directly from the data instead of relying on fixed heuristics. 

## How We Built It
Building CHOFormer involved several steps. We accessed a dataset of 97,000 CHO gene sequences from the NCBI database, focusing exclusively on protein-coding genes. These sequences are then filtered to retain those between 300 and 8000 base pairs, resulting in a refined dataset of 86,632 sequences. To reduce redundancy, `cd-hit-est` is employed to cluster the sequences based on an 8-word window and 90% nucleotide similarity, producing 47,713 sequences. The nucleotide sequences are then translated into their corresponding amino acid sequences, and any unnatural amino acids are removed to ensure biological relevance. The dataset is then split into training, validation, and test sets in an 80-10-10 ratio.

CHOFormer is built on the Transformer architecture, utilizing multiple decoder layers to map `ESM-2-150M` protein sequence embeddings to optimized codon sequences. To bridge the gap between amino acids and codon usage, we engineered a custom 3-mer tokenizer specifically for DNA sequences to accurately represent all codons.

To generate optimized codons, we project the ESM-2 embeddings into a higher-dimensional space before passing them through two decoder layers with four attention heads. Then, decoder logits are mapped to a probability distribution over our custom tokenizer's vocabulary to select optimized codons. With this approach, we generate DNA sequences with significantly improved protein yield and translational efficiency.


Parallelly, we developed CHOExp, a model designed to predict expression levels of the generated DNA sequences. CHOExp begins by accessing a dataset of 26,795 genes with corresponding RNA expression values. Genes with zero expression are removed, and the top 66% of genes that fall within three standard deviations are retained, resulting in a refined set of 13,253 genes. Expression values are then projected onto a log scale and normalized between 0 and 1 to allow sigmoid-based predictions. This dataset is split into training, validation, and test sets with an 80-10-10 split.

The core of CHOExp is an encoder-only transformer model with a dimensionality of 384, 8 layers, and 4 attention heads. The model is trained to predict protein expression levels based on the RNA expression data from the training set. CHOExp does not use any DNA foundation models as it's base, taking in the raw one-hot encoded vocab indices as input. Each DNA sequence is truncated/padded to a length of 1024 3-mer tokens (3072 total base pairs), and a classifier token <CLS> isa added at the start of the sequence. This input is processed through the transformer's attention and MLP processes. The output embedding of the <CLS> token is selected and processed to through a classification head, which consists of a linear layer and sigmoid activation function. After training the model on the training dataset for 10 epochs (including validation after every epoch), the expression model was evaluated on the test set and used to filter for high-expression CHO Genes when training CHOFormer.

## Challenges We Faced
One of the most significant challenges we faced was data preprocessing. The gene sequence data from public databases came in various formats and contained sequences that were incomplete, too short, or too long. Ensuring that the sequences we used were consistent and valid required extensive cleaning and filtering. Additionally, the clustering step using cd-hit-est was highly computationally expensive, and so we had to iteratively change our input (words and nucleotide similarity) to minimize compute time.

On the model side, training a Transformer decoder model to handle both protein embeddings and DNA sequence generation required careful architectural design. Handling long DNA sequences without overwhelming the model was a particular concern, as was ensuring that the model could generalize well to unseen sequences without overfitting to the training data.

## Evaluation
The Codon Adaptation Index (CAI) is a key metric used to predict gene expression efficiency based on codon usage, strongly correlating with real-world protein expression levels (dos Reis et al.). Similarly, the Translational Adaptation Index (TAI) measures how efficiently a sequence can be translated into protein, offering insights into translational efficiency. By applying Anwar et al.'s (2023) methodology for protein abundance prediction, we observed significant improvements after CHOFormer optimization.
The mean CAI of the optimized sequences was 0.8471 (± 0.0874), compared to the original mean CAI of 0.6541 (± 0.0526). Likewise, the mean TAI of the optimized sequences was 0.682 (± 0.209), compared to the original TAI of 0.373 (± 0.112). These results demonstrate substantial improvements in gene expression efficiency and translation potential using CHOFormer.












