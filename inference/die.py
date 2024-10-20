import requests
import json
from typing import List

def read_items_from_file(file_path: str) -> List[str]:
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

def chunk_list(lst: List[str], chunk_size: int) -> List[List[str]]:
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def make_api_request(url: str, sequences: List[str]) -> dict:
    payload = {"sequences": sequences}
    response = requests.post(url, json=payload)
    return response.json()

def save_response_to_file(response: List[dict], file_path: str):
    with open(file_path, 'w') as file:
        json.dump(response, file, indent=2)

def main():
    input_file = "protein_sequences.txt"
    output_file = "api_response.txt"
    api_url = "http://localhost:8000/choformer_inference"
    chunk_size = 4

    # Read items from file
    sequences = read_items_from_file(input_file)

    # Split sequences into chunks
    sequence_chunks = chunk_list(sequences, chunk_size)

    # Process chunks and collect results
    all_responses = []
    for i, chunk in enumerate(sequence_chunks):
        print(f"Processing chunk {i+1}/{len(sequence_chunks)}...")
        response = make_api_request(api_url, chunk)
        all_responses.extend(response)  # Assuming the API returns a list of results

    # Save collective response to file
    save_response_to_file(all_responses, output_file)

    print(f"All API responses saved to {output_file}")

if __name__ == "__main__":
    main()