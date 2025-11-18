from operator import getitem
from pathlib import Path
import torch
import pandas as pd
import glob
import librosa
from typing import List

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
        # index is the index of the file to load
        audio_path = self.wav_paths[index]
        nparray, sr = librosa.load(audio_path, sr=self.sample_rate)
        stem = Path(audio_path).stem
        if(stem.isnumeric()):
            label = int(stem)
        else:
            label = -1
        # convert nparray to tensor but print the shapes before returning
        print(f"Loading {audio_path} with shape {nparray.shape}")
        return torch.tensor(nparray), label

    @staticmethod
    def collate_fn(batch: List[tuple[torch.Tensor, int]]):
        audio_list = []
        label_list = []
        for batch_item in batch:
            audio_list.append(batch_item[0]) # always the audio data tensor
            label_list.append(batch_item[1]) # always the label

        # Pad sequences to the same length
        max_len = max([audio.shape[0] for audio in audio_list])
        padded_audios = []
        for audio in audio_list:
            if audio.shape[0] < max_len:
                padding = torch.zeros(max_len - audio.shape[0])
                padded_audio = torch.cat([audio, padding])
            else:
                padded_audio = audio  
            padded_audios.append(padded_audio)

        return torch.stack(padded_audios), torch.tensor(label_list)

# Get directory where this script is located
script_dir = Path(__file__).parent
audio_data = AudioData(glob.glob(str(script_dir / "audio" / "*.wav")))

dataloader = DataLoader(audio_data, batch_size=2, shuffle=True, collate_fn=AudioData.collate_fn)

for batch in dataloader:
    print(batch)

if __name__ == "__main__":
    print(audio_data.wav_paths)