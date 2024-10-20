import csv

input_file = "protein_exp.csv"
output_file = "processed_proteinexp.csv"

columns_to_extract = [
    "Accession ID", "Protein Names", "GO ID", "GO Slim Function", 
    "Swiss-Prot Annotation", "RNA-seq mapping depth", "# AAs", 
    "Spectra Number", "SAF", "NSAF", "Status", "p-value"
]

def extract_accession_id(full_accession):
    """Extracts just the accession ID (e.g., XP_003494975.1) from the full string"""
    parts = full_accession.split('|')
    if len(parts) >= 4:
        return parts[3]  # Extracts the part after 'ref|'
    return None

with open(input_file, mode="r", newline='', encoding="utf-8") as infile:
    reader = csv.reader(infile)
    
    with open(output_file, mode="w", newline='', encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        
        writer.writerow(columns_to_extract)
        
        next(reader)
        
        for row in reader:
            accession_id = extract_accession_id(row[0]) 
            protein_names = row[1]
            go_id = row[2]
            go_slim_function = row[3]
            swiss_prot_annotation = row[4]
            rna_seq_mapping_depth = row[5]
            num_aa = row[6]
            spectra_number = row[7]
            saf = row[8]
            nsaf = row[9]
            status = row[11]
            p_value = row[12] 
            
            writer.writerow([
                accession_id, protein_names, go_id, go_slim_function, swiss_prot_annotation, 
                rna_seq_mapping_depth, num_aa, spectra_number, saf, nsaf, status, p_value
            ])

print(f"Processed data saved to {output_file}")
