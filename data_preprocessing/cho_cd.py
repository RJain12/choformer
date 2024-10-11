import os
import subprocess

def rename_to_fasta(file_path):
    base, ext = os.path.splitext(file_path)
    if ext.lower() == '.fna':
        new_file_path = base + '.fasta'
        os.rename(file_path, new_file_path)
        print(f"File renamed to {new_file_path}")
        return new_file_path
    return file_path

def run_cd_hit_est(input_fasta, output_fasta, similarity=0.95):
    cd_hit_cmd = [
        "cd-hit-est",
        "-i", input_fasta,
        "-o", output_fasta,
        "-c", str(similarity), 
        "-n", "8",
        "-d", "0" 
    ]

    try:
        subprocess.run(cd_hit_cmd, check=True)
        print(f"CD-HIT-EST successfully run. Output written to {output_fasta}")
    except subprocess.CalledProcessError as e:
        print(f"Error running CD-HIT-EST: {e}")

if __name__ == "__main__":
    # Path to your file
    input_file = "GCA_003668045.2_CriGri-PICRH-1.0_genomic.fna"
    
    # Step 1: Rename to .fasta if necessary
    fasta_file = rename_to_fasta(input_file)
    
    # Step 2: Run CD-HIT-EST
    output_file = "clustered_output.fasta"
    run_cd_hit_est(fasta_file, output_file)
