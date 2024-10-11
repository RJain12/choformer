1. Download from https://www.ncbi.nlm.nih.gov/datasets/gene/taxon/10029/?gene_type=protein-coding 
(Protein coding genes only, 22956 genes downloaded)
2. 150 < bp < 6000:
`seqkit seq -m 150 -M 6000 input.fasta -o filtered_output.fasta`
3. cd-hit-est:
`cd-hit-est -i filtered_output.fasta -o output_clusters.fa -c 0.95 -n 8 -T 8 -M 20000`
