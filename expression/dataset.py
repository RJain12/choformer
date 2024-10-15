from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
from functools import partial


class RNASeqCore(Dataset):
    def __init__(self, data_path: str):
        super().__init__()

        # opening the thing
        with open(data_path, "r") as f:
            values = f.read().split("\n")[1:]
            self.data = [(x.split(",")[0], float(x.split(",")[1])) for x in values]
            # self.data = []
            # for x in values:
            #     print(x.split(","))
            #     self.data.append((x.split(",")[0], float(x.split(",")[1])))
            
        # self.seqs = []
        # intron_idxs = []
        # for ix, seq_name in tqdm(enumerate([dat[0] for dat in self.data]), desc="Extracting sequence data"):
        #     try:
        #         with open(f"{fasta_path}/{seq_name}.fasta", "r") as f:
        #             self.seqs.append("".join(f.read().split("\n")[1:]))
        #     except:
        #         # print(f"fasta sequence not found ({seq_name})")
        #         intron_idxs.append(ix)
        #         # raise KeyError(f"fasta not found ({seq_name})")
        
            
        # for i in reversed(range(len(intron_idxs))):
        #     del self.data[i]
        
        
            
    def __len__(self): return len(self.data)
    
    def __getitem__(self, ix: int):
        return self.data[ix]


def collate(batch: list[tuple[str, float]], tokenizer, max_length: int):
    y = torch.as_tensor([b[1] for b in batch])
    X = tokenizer([b[0] for b in batch], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt").input_ids
    
    return X, y


class RNASeq:
    def __init__(self, train_data_path: str, val_data_path: str, batch_size: int, tokenizer_path: str, max_length: int, num_workers: int = 8):
        self.train = RNASeqCore(train_data_path)
        self.val = RNASeqCore(val_data_path)
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            
        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=partial(collate, max_length=self.max_length, tokenizer=self.tokenizer))
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=partial(collate, max_length=self.max_length, tokenizer=self.tokenizer))
    


# if __name__ == "__main__":
#     rnaseq = RNASeq()
    
#     rnaseq[0]
             