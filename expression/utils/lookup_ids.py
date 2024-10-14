import os
import pandas as pd
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Number of retries on failure
MAX_RETRIES = 2
# Delay factor for retries
RETRY_DELAY = 5  # seconds

# Function to fetch FASTA data for a single ID
def fetch_fasta(id, gb_acc_value, fasta_file):
    # Check if the file already exists
    if os.path.exists(fasta_file) and os.path.getsize(fasta_file) > 0:
        print(f"FASTA file already exists for ID: {id}. Skipping download.")
        # Return the existing content
        with open(fasta_file, "r") as f:
            return f.read().replace('\n', '')
    
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            # Use efetch to fetch FASTA sequence and capture the result in a temporary variable
            print(f"Fetching FASTA for GB_ACC: {gb_acc_value} (Attempt {attempt + 1})")
            result = subprocess.run(
                ["efetch", "-db", "nuccore", "-id", gb_acc_value, "-format", "fasta"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )

            # If successful, write the result to a file only if it's non-empty
            if result.stdout and len(result.stdout) > 0:
                with open(fasta_file, "w") as f:
                    f.write(result.stdout.decode('utf-8'))
                print(f"FASTA data saved for ID: {id}")
                # Return the fetched FASTA sequence
                return result.stdout.decode('utf-8').replace('\n', '')
            else:
                print(f"Empty response for ID: {id}, retrying...")
        
        except subprocess.CalledProcessError as e:
            print(f"Error fetching FASTA for ID {id}: {e.stderr.decode('utf-8')}")
        
        # Increment attempt counter
        attempt += 1
        # Sleep before retrying to avoid overwhelming the server
        time.sleep(RETRY_DELAY * attempt)
    
    print(f"Failed to fetch FASTA for ID: {id} after {MAX_RETRIES} attempts.")
    return None

# Process RNA data in parallel
def process_rna_data():
    rna = pd.read_csv(os.path.join('..', 'data', "rna", 'rna.csv'))
    lookup = pd.read_csv(os.path.join('..', 'data', "rna", 'rna_lookup.txt'), sep='\t')
    
    # Directory to store the fetched FASTA sequences temporarily
    fasta_dir = os.path.join("..", 'data', "rna", "rna_expression_seqs")
    if not os.path.exists(fasta_dir):
        os.makedirs(fasta_dir)

    # Create a thread pool for parallel execution
    with ThreadPoolExecutor(max_workers=5) as executor:  # Reduced max_workers to prevent overloading the server
        futures = []
        
        # Loop over each row in the RNA data
        for index, row in rna.iterrows():
            id = row['id']
            print(f"Processing ID: {id}")
            
            # Find the corresponding GB_ACC from the lookup file
            lookup_row = lookup[lookup['ID'] == id]
            gb_acc = lookup_row['GB_ACC'].values

            # if it's an intron, skip it
            if (lookup_row["Gene Symbol"] == "INTRON").bool():
                print(f"Skipping ID: {id} as it's an intron")
                continue
            
            if len(gb_acc) > 0:
                gb_acc_value = str(gb_acc[0])
                fasta_file = os.path.join(fasta_dir, f"{id}.fasta")  # Path to the FASTA file

                # Submit a task to fetch FASTA for each ID
                futures.append(executor.submit(fetch_fasta, id, gb_acc_value, fasta_file))
            else:
                print(f"No GB_ACC found for ID: {id}")
        
        # Process the completed futures as they finish
        for index, future in enumerate(as_completed(futures)):
            try:
                fasta_sequences = future.result()
                if fasta_sequences:
                    rna.at[index, 'fasta'] = fasta_sequences
            except Exception as e:
                print(f"An error occurred during processing: {e}")

    # Save the updated RNA data with the FASTA sequences
    rna.to_csv(os.path.join('..', 'data', "rna", 'rna_with_fasta.csv'), index=False)
    print("Finished processing RNA data and saved to 'rna_with_fasta.csv'")

# Main function to run the RNA processing
def main():
    process_rna_data()

if __name__ == "__main__":
    main()