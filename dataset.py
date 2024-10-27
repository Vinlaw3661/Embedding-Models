class EncoderDataset(torch.utils.data.Dataset):
    def __init__(self, datapath):
        self.data = pd.read_csv(datapath, sep='\t', nrows=300)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        return self.data.iloc[idx]['questions'], self.data.iloc[idx]['answers']