import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
import sentencepiece as spm
import os

# Text corpus to be used
text_corpus = open("data.txt", "r").read()

# SentencePiece tokenizer training
def train_tokenizer(text, model_prefix='m', vocab_size=5000):
    with open('text.txt', 'w') as f:
        f.write(text)
    spm.SentencePieceTrainer.Train(f'--input=text.txt --model_prefix={model_prefix} --vocab_size={vocab_size}')
    sp = spm.SentencePieceProcessor()
    sp.load(f'{model_prefix}.model')
    return sp

#sp = train_tokenizer(text_corpus)

# Initialize a SentencePieceProcessor object
sp = spm.SentencePieceProcessor()

# The filename of your previously trained SentencePiece model
model_filename = 'm.model'

# Load the model file into the SentencePieceProcessor
sp.Load(model_filename)


tokenized_text = sp.encode(text_corpus)

dataset = [(tokenized_text[i], tokenized_text[i + 1]) for i in range(len(tokenized_text) - 1)]
test_size = int(len(dataset)*.25)
print (f"test_size = {test_size}")
test_dataset = dataset[-test_size:]
dataset = dataset[0:-test_size]

# Defining the model
class FFNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.hebbian = nn.Linear(hidden_size, hidden_size, bias=False)
        #self.hebbian.requires_grad = False
        self.fc2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        t = 1.0 * x
        self.x = x
        x = self.hebbian(x)
        #self.hebbian.weight = torch.nn.Parameter(l * self.hebbian.weight + (1 - l) * torch.outer(t.squeeze(), t.squeeze()))
        x = torch.sigmoid(x+t)
        x = self.fc2(x)
        
        return x

# Parameters
VOCAB_SIZE = sp.GetPieceSize()
EMBED_SIZE = 128
HIDDEN_SIZE = 128
LR = 0.01
EPOCHS = 1000
l = 0.5

model = FFNN(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# Training the model
def train(model, dataset, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        count = 0
        for i, (input_word, target_word) in enumerate(dataset):
            input_word = torch.tensor([input_word])
            target_word = torch.tensor([target_word])

            output = model(input_word)
            loss = criterion(output, target_word)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1

            if (i+1) % 10000 == 0:
                print(f'Epoch: {epoch} Token: {i}/{len(dataset)} Loss: {total_loss/count}', flush=True)

        print(f'Epoch: {epoch} Loss: {total_loss/count}', flush=True)
        # save model checkpoint

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss/count,
        }, f'checkpoint_{epoch}.pth')

# Function for prediction
def predict_next_word(tok, model, sp):
    model.eval()
    with torch.no_grad():
        input_word = torch.tensor([tok])
        output = model(input_word)
        _, predicted = torch.max(output, 1)
        return predicted.item()

def load_checkpoint(model, optimizer, filename):
    # Note: Input model & optimizer should be pre-defined. This routine only updates their states.
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, loss

#model, optimizer, start_epoch, loss = load_checkpoint(model, optimizer, 'checkpoint_48.pth')
train(model, dataset, EPOCHS)


# Example of prediction
for tok in dataset[0:100]:
    predicted_word = predict_next_word(tok[0], model, sp)
    print (sp.IdToPiece(tok[1]), sp.IdToPiece(predicted_word))

