import torch
import torch.nn as nn
import torch.optim as optim
import random

translation_pairs = [
    ("i am a student", "je suis un étudiant"),
    ("he is a teacher", "il est un professeur"),
    ("she is happy", "elle est heureuse"),
    ("they are playing", "ils jouent"),
    ("you are smart", "tu es intelligent")
]

def tokenize(sentence):
    return sentence.lower().split()

def build_vocabulary(sentences):
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
    index = 3
    for sentence in sentences:
        for word in tokenize(sentence):
            if word not in vocab:
                vocab[word] = index
                index += 1
    return vocab

source_vocab = build_vocabulary([src for src, _ in translation_pairs])
target_vocab = build_vocabulary([tgt for _, tgt in translation_pairs])
reverse_target_vocab = {v: k for k, v in target_vocab.items()}

def encode(sentence, vocab):
    return [vocab["<sos>"]] + [vocab[word] for word in tokenize(sentence)] + [vocab["<eos>"]]

training_data = [(encode(src, source_vocab), encode(tgt, target_vocab)) for src, tgt in translation_pairs]

SRC_VOCAB_SIZE = len(source_vocab)
TGT_VOCAB_SIZE = len(target_vocab)
EMBED_SIZE = 32
HIDDEN_SIZE = 64

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden):
        input = input.unsqueeze(0)
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc(output.squeeze(0))
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        tgt_len = tgt.shape[0]
        tgt_vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(tgt_len, tgt_vocab_size)
        hidden = self.encoder(src)
        input = tgt[0]
        for t in range(1, tgt_len):
            output, hidden = self.decoder(input, hidden)
            outputs[t] = output
            top1 = output.argmax(1)
            input = tgt[t] if random.random() < teacher_forcing_ratio else top1
        return outputs

encoder = Encoder(SRC_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE)
decoder = Decoder(TGT_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE)
model = Seq2Seq(encoder, decoder)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=0)

for epoch in range(100):
    total_loss = 0
    for src, tgt in training_data:
        src_tensor = torch.tensor(src).unsqueeze(1)
        tgt_tensor = torch.tensor(tgt).unsqueeze(1)
        optimizer.zero_grad()
        output = model(src_tensor, tgt_tensor)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        tgt_tensor = tgt_tensor[1:].view(-1)
        loss = criterion(output, tgt_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

def translate(model, sentence, max_len=10):
    model.eval()
    tokens = encode(sentence, source_vocab)
    src_tensor = torch.tensor(tokens).unsqueeze(1)
    hidden = model.encoder(src_tensor)
    input = torch.tensor([target_vocab["<sos>"]])
    result = []
    for _ in range(max_len):
        output, hidden = model.decoder(input, hidden)
        top1 = output.argmax(1).item()
        if top1 == target_vocab["<eos>"]:
            break
        result.append(reverse_target_vocab[top1])
        input = torch.tensor([top1])
    return ' '.join(result)

print()
print("Input: 'i am a student'")
print("Output:", translate(model, "i am a student"))
print("Input: 'you are smart'")
print("Output:", translate(model, "you are smart"))
