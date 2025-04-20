import argparse, os, random, itertools, json
from collections import Counter
import pandas as pd
import torch, torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torchtext.data.utils import get_tokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.utilities import PreprocessingUtils

# Reproduceability
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Vocabulary
def build_vocab(tokenizer, texts, min_freq=2, max_size=30000):
    counter = Counter(itertools.chain.from_iterable(texts.apply(tokenizer)))
    specials = ["<pad>", "<unk>"]
    # Keep words meeting min_freq and under max_size
    most_common = [word for word, freq in counter.items() if freq >= min_freq]
    most_common = most_common[: max_size - len(specials)]
    stoi = {tok: idx for idx, tok in enumerate(specials + most_common)}
    itos = {idx: tok for tok, idx in stoi.items()}
    PAD_IDX, UNK_IDX = stoi["<pad>"], stoi["<unk>"]
    return stoi, itos, PAD_IDX, UNK_IDX

# Define a custom Dataset and collate function
class DisasterTweetsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, stoi, unk_idx):
        self.seqs = [
            torch.tensor([stoi.get(tok, unk_idx) for tok in tokenizer(txt)],
                         dtype=torch.long)
            for txt in texts
        ]
        self.labels = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx): return self.seqs[idx], self.labels[idx]

def collate_pad(batch, pad_idx):
    seqs, labels = zip(*batch)
    lens = torch.tensor([len(s) for s in seqs])
    seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=pad_idx)
    return seqs_padded, lens, torch.tensor(labels)

# Define model structure
class TweetClassifier(nn.Module):
    def __init__(self, vocab_sz, embed_dim, hidden_dim,
                 num_layers, pad_idx, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_sz, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=bidirectional, dropout=dropout,
                            batch_first=True)
        lstm_out = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_out, 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, lens):
        emb = self.drop(self.embedding(x))
        packed = pack_padded_sequence(emb, lens.cpu(),
                                      batch_first=True, enforce_sorted=False)
        output, (hn, _) = self.lstm(packed)
        if self.lstm.bidirectional:
            h = torch.cat((hn[-2], hn[-1]), dim=1)
        else:
            h = hn[-1]
        return self.fc(self.drop(h))

# Define training and evaluation functions
def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    tot_loss = tot_corr = n = 0
    for x, lens, y in loader:
        x, lens, y = x.to(device), lens.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x, lens)
        loss = loss_fn(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        tot_loss += loss.item() * y.size(0)
        tot_corr += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return tot_loss / n, tot_corr / n

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    tot_loss = tot_corr = n = 0
    for x, lens, y in loader:
        x, lens, y = x.to(device), lens.to(device), y.to(device)
        logits = model(x, lens)
        loss = loss_fn(logits, y)
        tot_loss += loss.item() * y.size(0)
        tot_corr += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return tot_loss / n, tot_corr / n

# Define prediction function
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for x, lens in loader:
        x, lens = x.to(device), lens.to(device)
        logits = model(x, lens)
        preds.extend(logits.argmax(1).cpu().tolist())
    return preds

# Define main function to handle command line arguments and execute training or prediction
def main():
    parser = argparse.ArgumentParser(
        description="Train or predict a disaster-tweet classifier.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train_p = sub.add_parser("train")
    train_p.add_argument("--train_csv", required=True)
    train_p.add_argument("--model_dir", default="model")
    train_p.add_argument("--epochs", type=int, default=10)

    pred_p = sub.add_parser("predict")
    pred_p.add_argument("--test_csv", required=True)
    pred_p.add_argument("--model_dir", default="model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer("basic_english")

    # Model training section
    if args.cmd == "train":
        print("Starting training...")
        # Load and preprocess training data
        df = pd.read_csv(args.train_csv)
        df = PreprocessingUtils.clean_keywords(df)
        df['combined'] = (df.apply(PreprocessingUtils.combine_text, axis=1)
                            .apply(PreprocessingUtils.clean_text))
        print(f"Dataset loaded and preprocessed with {len(df)} samples.")

        train_df, val_df = train_test_split(
            df, test_size=0.2, stratify=df['target'], random_state=SEED)
        print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")

        # Build vocabulary
        stoi, itos, PAD_IDX, UNK_IDX = build_vocab(tokenizer, df['combined'])
        os.makedirs(args.model_dir, exist_ok=True)
        json.dump({"itos": itos}, open(os.path.join(args.model_dir, "vocab.json"), "w"))
        print(f"Vocabulary built with size {len(stoi)}.")

        def make_loader(sub_df, batch, shuffle):
            ds = DisasterTweetsDataset(
                    sub_df['combined'], sub_df['target'],
                    tokenizer, stoi, UNK_IDX)
            return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                              collate_fn=lambda b: collate_pad(b, PAD_IDX),
                              drop_last=False)
        
        train_dl = make_loader(train_df, 64, True)
        val_dl   = make_loader(val_df,   128, False)

        # Build and train the model
        model = TweetClassifier(len(stoi), embed_dim=200, hidden_dim=128,
                                num_layers=2, pad_idx=PAD_IDX).to(device)
        opt      = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
        scheduler= ReduceLROnPlateau(opt, factor=0.5, patience=1)
        loss_fn  = nn.CrossEntropyLoss()

        best_acc = 0
        for epoch in range(1, args.epochs+1):
            tr_loss, tr_acc = train_epoch(model, train_dl, opt, loss_fn, device)
            val_loss, val_acc = evaluate(model, val_dl, loss_fn, device)
            scheduler.step(val_loss)

            print(f"[{epoch:02d}] "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                  f"val loss {val_loss:.4f} acc {val_acc:.3f}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    "model_state": model.state_dict(),
                    "pad_idx": PAD_IDX,
                    "unk_idx": UNK_IDX,
                }, os.path.join(args.model_dir, "bilstm.pt"))

        print(f"Model training completed with best validation accuracy: {best_acc:.3f}")

    # Prediction section
    else:
        print("Starting prediction...")
        # Load the model and vocabulary
        vocab = json.load(open(os.path.join(args.model_dir, "vocab.json")))
        stoi  = {tok: int(idx) for idx, tok in vocab["itos"].items()}
        PAD_IDX = stoi["<pad>"]; UNK_IDX = stoi["<unk>"]

        ckpt = torch.load(os.path.join(args.model_dir, "bilstm.pt"),
                          map_location=device)
        model = TweetClassifier(len(stoi), embed_dim=200, hidden_dim=128,
                                num_layers=2, pad_idx=PAD_IDX).to(device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Model and artifacts loaded from {args.model_dir} directory.")

        # Load and preprocess test data
        test_df = pd.read_csv(args.test_csv)
        test_df = PreprocessingUtils.clean_keywords(test_df)
        test_df['combined'] = (test_df.apply(PreprocessingUtils.combine_text, axis=1)
                                   .apply(PreprocessingUtils.clean_text))
        print(f"Test dataset loaded and preprocessed with {len(test_df)} samples.")

        test_ds = [
            torch.tensor([stoi.get(tok, UNK_IDX) for tok in tokenizer(txt)],
                         dtype=torch.long)
            for txt in test_df['combined']
        ]
        test_dl = DataLoader(test_ds, batch_size=128, shuffle=False,
                             collate_fn=lambda batch:
                                 (pad_sequence(batch, batch_first=True, padding_value=PAD_IDX),
                                  torch.tensor([len(x) for x in batch])))

        preds = predict(model,
                        ((x, lens) for x, lens in test_dl),
                        device)
        
        pd.DataFrame({"id": test_df["id"], "target": preds}) \
          .to_csv("predictions.csv", index=False)
        print("Saved predictions.csv")

if __name__ == "__main__":
    main()
