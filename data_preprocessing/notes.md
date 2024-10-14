## New Approach
1. Download from https://www.ncbi.nlm.nih.gov/datasets/gene/taxon/10029/?gene_type=protein-coding 
(Protein coding genes only, 22956 genes downloaded, approximately 43000 sequences)
2. Use `splitfasta.py` to break it up into 10 segments (~4300 sequences each)
3. cd-hit-est for each segment:
`cd-hit-est -i output_parts/gene_part_1.fasta -o output_parts/output_gene_part_1.fasta -c 0.95 -n 10 -T 8 -M 30000`
4. Use `combinefasta_2.py` to combine `output_gene_part_1.fasta` and `output_gene_part_2.fasta` and so on to output `combine_genepart_1_2.fasta`
5. cd-hit-est for the five merged files to output `output_combine_genepart_1_2.fasta`
6. Use `combinefasta_3-2.py` to combine `output_combine_genepart_1_2.fasta` and `output_combine_genepart_3_4.fasta` and `output_combine_genepart_5_6.fasta` into one file, and the remaining two into another called `combine_gene123456` and `combine_gene78910`
7. Run cd-hit-est on the two merged files to output `output_combine_gene123456`and `output_combine_gene78910`
8. Merge to form `final.fasta`
9. cd-hit-est 
## Old Approach
1. Download from https://www.ncbi.nlm.nih.gov/datasets/gene/taxon/10029/?gene_type=protein-coding 
(Protein coding genes only, 22956 genes downloaded)
2. 150 < bp < 6000:
`seqkit seq -m 150 -M 6000 input.fasta -o filtered_output.fasta`
3. cd-hit-est:
`cd-hit-est -i filtered_output.fasta -o output_clusters.fa -c 0.95 -n 8 -T 8 -M 20000`
YIELDED ~8000 GENES
