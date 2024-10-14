import os
from huggingface_hub import login
from typing import Union, List
import torch
from esm.models.esm3 import ESM3
import glob
from esm.sdk.api import ESM3InferenceClient, ESMProtein, SamplingConfig, ESMProteinTensor
import esm

"""
ESM Wrapper for easy protein embedding generation

Before using this wrapper, you need to obtain an ESM Forge token:
(message Tyler or do the following:)
1. Go to https://forge.evolutionaryscale.ai/sign-up
2. Log in with the account
3. Navigate to API section
4. Copy API token
5. Set the token as an environment variable:
   - For Unix/Linux/macOS: export ESM_FORGE_TOKEN=your_token_here
   - For Windows: setx ESM_FORGE_TOKEN your_token_here

For local use, you will need to log into Hugging Face (how to login: https://huggingface.co/docs/huggingface_hub/en/quick-start):

"""

class ESMWrapper:
    def __init__(self, model_name: str = "esm3-medium-2024-08", use_local: bool = False, device: str = "cuda"):
        """
        Initialize the ESM Wrapper.

        Args:
            model_name (str): Name of the ESM model to use.
            use_local (bool): Whether to use a local model or the API.
            device (str): Device to use for local models ('cuda' or 'cpu').

        Raises:
            ValueError: If ESM_FORGE_TOKEN is not set when using the API.
        """

        self.use_local = use_local
        self.device = device

        if use_local:
            self.client = ESM3.from_pretrained(model_name).to(device)
        else:
            token = os.environ.get("ESM_FORGE_TOKEN")
            if token is None:
                raise ValueError("ESM_FORGE_TOKEN environment variable is not set")
            self.client = esm.sdk.client(model_name, token=token)

    def encode_protein(self, sequence: Union[str, List[str]]) -> Union[ESMProteinTensor, List[ESMProteinTensor]]:
        if isinstance(sequence, str):
            protein = ESMProtein(sequence=sequence)
            return self.client.encode(protein)
        elif isinstance(sequence, list):
            proteins = [ESMProtein(sequence=seq) for seq in sequence]
            return [self.client.encode(protein) for protein in proteins]
        else:
            raise ValueError("Input must be a string or a list of strings")

    def get_embeddings(self, protein: Union[str, List[str]], return_dict: bool = False) -> List[List[List[float]]]:
        """
        Get embeddings for a protein or a list of proteins.

        Args:
            protein - a string or list of strings
            return_dict - if True, return a dictionary where keys are protein sequence strings

        Returns:
            List of embeddings for each protein.
        """

        if isinstance(protein, (str, list)):
            protein = self.encode_protein(protein)
        
        sampling_config = SamplingConfig(return_per_residue_embeddings=True)
        
        if isinstance(protein, list):
            outputs = [self.client.forward_and_sample(p, sampling_config) for p in protein]
        else:
            outputs = [self.client.forward_and_sample(protein, sampling_config)]
        
        embeddings = [output.per_residue_embedding for output in outputs]
        
        if return_dict:
            if isinstance(protein, list):
                protein_strings = [p.sequence for p in protein]
            else:
                protein_strings = [protein.sequence]
            
            return dict(zip(protein_strings, embeddings))
        else:
            return embeddings

    def get_embeddings_from_pdb(self, pdb_path: str) -> Union[List[List[List[float]]], dict]:
        """
        Get embeddings for a proteins from a PDB file or directory of PDB files.

        Args:
            pdb_path (str): Path to the PDB file or directory with PDB files inside.

        Returns:
            List of embeddings for the protein if pdb_path is a file, or a dictionary with PDB file names as keys and lists of embeddings as values if pdb_path is a directory.
        """

        if os.path.isfile(pdb_path):
            protein = ESMProtein.from_pdb(pdb_path)
            protein_tensor = self.client.encode(protein)
            return self.client.forward_and_sample(protein_tensor, SamplingConfig(return_per_residue_embeddings=True))
        elif os.path.isdir(pdb_path):
            pdb_files = glob.glob(os.path.join(pdb_path, "*.pdb"))
            embeddings_dict = {}

            for pdb_file in pdb_files:
                protein_name = os.path.basename(pdb_file).split(".")[0]
                embeddings_dict[protein_name] = self.get_embeddings_from_pdb(pdb_file)

            return embeddings_dict
        else:
            raise ValueError("pdb_path must be a file or a directory")



# Usage example:
if __name__ == "__main__":
    # For API usage
    esm_api = ESMWrapper(model_name="esm3-medium-2024-08", use_local=False)
    
    # For local usage
    # esm_local = ESMWrapper(model_name="esm3_sm_open_v1", use_local=True, device="cuda")

    # Single sequence
    sequence = "FIFLALLGAAVAFPVDDDDKIVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQWVVSAAHCYKSGIQVRLGEDNINVVEG"
    
    # Get embeddings for a single sequence
    embeddings = esm_api.get_embeddings(sequence)
    print(f"Single sequence embeddings shape: {len(embeddings)} x {len(embeddings[0])} x {len(embeddings[0][0])}")

    # Multiple sequences
    sequences = [
        "FIFLALLGAAVAFPVDDDDKIVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQWVVSAAHCYKSGIQVRLGEDNINVVEG",
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    ]

    # Get embeddings for multiple sequences
    multi_embeddings = esm_api.get_embeddings(sequences, return_dict=True)
