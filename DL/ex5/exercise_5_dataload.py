import torch
import pandas as pd
import glob
import librosa

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class AudioData(Dataset):
    sample_rate = 16000

    # Discover and Store File Paths
    def __init__(self, wav_paths):
        super().__init__()    
        self.wav_paths = wav_paths

    def __len__(self):
        return len(self.wav_paths)

    # Load one Wave file to a tensor
    def __getitem__(self, index):
        audio_path = self.wav_paths[index]
        
        
        

    # use DataLoader to iterate over those items
    # with custom collate_fn


class Runner:
    audio_data = AudioData(glob.glob("audio/*.wav", recursive=True))

    print(audio_data.wav_paths)