import torch
import torch.nn as nn
from dataset import get_dataloader
from model import Seq2Seq

def build_vocab():
    # map letters to number 0-25
    char_to_idx = {chr(ord('a') + i): i for i in range(26)}
    # add special tokens
    char_to_idx['<PAD>'] = 26 #pad to the same length
    char_to_idx['<SOS>'] = 27 # start of seq
    char_to_idx['<EOS>'] = 28 # end of seq
    return char_to_idx

def encode_string(s, char_to_idx, max_len):
    encoded = [char_to_idx['<SOS>']]
    for char in s:
        encoded.append(char_to_idx[char])
    encoded.append(char_to_idx['<EOS>'])
    while len(encoded) < max_len:
        encoded.append(char_to_idx['<PAD>'])
    return torch.tensor(encoded[:max_len])

def train(model, data, optimizer, criterion, epochs, checkpoint_path):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        # we use only one sample in one batch
        # batch size is 1
        for src, trg in data:
            optimizer.zero_grad()
            src = src.unsqueeze(0)
            trg = trg.unsqueeze(0)
            prediction = model(src, trg)
            loss = criterion(prediction.view(-1, prediction.size(-1)), trg.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data):.4f}")
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'use_attention': model.use_attention}, checkpoint_path)

if __name__ == "__main__":
    csv_file = "toy_dataset.csv"
    dataloader = get_dataloader(csv_file, batch_size=1, shuffle=False)
    
    data = []
    for inputs, outputs in dataloader:
        # input: ('abc',)
        # output: ('cba',)
        # data.append(('abc', 'cba'))
        data.append((inputs[0], outputs[0]))
    
    char_to_idx = build_vocab()
    vocab_size = len(char_to_idx)
    
    max_len = 15 # pad to max
    encoded_data = []
    for inputs, outputs in data:
        src = encode_string(inputs, char_to_idx, max_len)
        trg = encode_string(outputs, char_to_idx, max_len)
        encoded_data.append((src, trg))
    
    use_attention = True 
    model = Seq2Seq(vocab_size, embedding_dim=64, hidden_dim=128, num_layers=1, use_attention=use_attention)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=char_to_idx['<PAD>'])
    
    epochs = 5
    checkpoint_path = "checkpoint.pt"
    train(model, encoded_data, optimizer, criterion, epochs, checkpoint_path)

