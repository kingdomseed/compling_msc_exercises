import torch
from dataset import get_dataloader
from model import Seq2Seq
from train import build_vocab, encode_string

def test(input_string, checkpoint_path, max_len=15):
    char_to_idx = build_vocab()
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    vocab_size = len(char_to_idx)
    
    checkpoint = torch.load(checkpoint_path)
    use_attention = checkpoint.get('use_attention', True)
    
    model = Seq2Seq(vocab_size, embedding_dim=64, hidden_dim=128, num_layers=1, use_attention=use_attention)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    src = encode_string(input_string, char_to_idx, max_len).unsqueeze(0)
    with torch.no_grad():
        pred_indices = model.predict(src, max_len, char_to_idx['<SOS>'], char_to_idx['<EOS>'])
    
    prediction = ''.join([idx_to_char[idx] for idx in pred_indices if idx_to_char[idx] not in ['<SOS>', '<EOS>', '<PAD>']])
    return prediction

if __name__ == "__main__":
    result = test("hello", "checkpoint.pt")
    print(result)

