import torch
import torch.nn as nn
import numpy as np
import re
from nltk.tokenize import word_tokenize



device = 'cuda' if torch.cuda.is_available else 'cpu'
print(f"Model set to use = ",device)

#=============================== Model Defination ============================

class GoEmotions_lstm(nn.Module):
    def __init__(self, vocab_size, embed_dim=200, hidden_dim=256, num_classes=28, nun_layers=2):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_dim,
                            num_layers=nun_layers,
                            batch_first=True,
                            dropout=0.2,
                            bidirectional=True)
        
        self.fc = nn.Linear(hidden_dim*2 , num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.embeddings(x)

        out, (h, c) = self.lstm(x)
        h_forward = h[-2]
        h_backward = h[-1]

        h_cat = torch.cat((h_forward, h_backward), dim=1)

        h_cat = self.dropout(h_cat)

        out = self.fc(h_cat)

        return out
    
#========================== Loading Model ====================================

def load_goemotion_model(path="goemotions_bilstm_checkpoint.pth"):
    checkpoint = torch.load(path, map_location="cpu")

    vocab = checkpoint["vocab"]
    max_len = checkpoint["max_len"]
    threshold = checkpoint["threshold"]

    model = GoEmotions_lstm(vocab_size=len(vocab))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, vocab, max_len, threshold

#============================= Prediction Function ===================================

emotion_map = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion",
    "curiosity","desire","disappointment","disapproval","disgust","embarrassment",
    "excitement","fear","gratitude","grief","joy","love","nervousness","optimism",
    "pride","realization","relief","remorse","sadness","surprise","neutral"
]

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_emotions(text, model, vocab, max_len, threshold=0.3):

    model.eval()

    # preprocess
    text = clean_text(text)
    tokens = word_tokenize(text)

    # convert to indices
    seq = [vocab.get(tok, 1) for tok in tokens]   # <UNK>=1

    # pad / truncate
    if len(seq) < max_len:
        seq += [vocab["<PAD>"]] * (max_len - len(seq))
    else:
        seq = seq[:max_len]

    # tensor
    x = torch.tensor([seq], dtype=torch.long)

    # move input to same device as model
    device = next(model.parameters()).device
    x = x.to(device)

    # forward pass
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    # sort by probability
    sorted_idx = np.argsort(probs)[::-1]   # highest first

    # top predicted emotion
    top_idx = sorted_idx[0]
    top_emotion = emotion_map[top_idx]
    top_prob = float(probs[top_idx])

    # next 5 emotions
    others = []
    for idx in sorted_idx[1:6]:
        others.append((emotion_map[idx], float(probs[idx])))

    return {
        "top": (top_emotion, top_prob),
        "others": others
    }


model, vocab, max_len, threshold = load_goemotion_model()

#============================== test model ===================================

def find_emotions(text):
    return predict_emotions(text, model, vocab, max_len, threshold)

