import torch
import torch.nn.functional as F
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import time

def prepare_data(filename):
    df = pd.read_csv(filename)
    headlines = df['headline'][:2000].tolist()
    tokens = [['.'] + headline.split() + ['.'] for headline in headlines]
    word_counts = Counter([word for headline in tokens for word in headline])
    vocab = {word: i for i, word in enumerate(word_counts.keys())}
    vocab_size = len(vocab)
    token_ids = [[vocab[word] for word in headline] for headline in tokens]
    print("Data prepared!! Found ",vocab_size," new words")
    return token_ids, vocab, vocab_size


def train_test_split_data(token_ids, test_size=0.2, random_state=42):
    train_data, test_data = train_test_split(token_ids, test_size=test_size, random_state=random_state)
    print("Data split into train(", len(train_data), ") and test(", len(test_data), ")")
    return train_data, test_data


def TrainData(train_data, epochs, learning_rate):
    xs, ys = [], []
    for headline in train_data:
        for word1, word2 in zip(headline, headline[1:]):
            xs.append(word1)
            ys.append(word2)
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    num = xs.nelement()

    g = torch.Generator().manual_seed(2147483647)
    # W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)
    W = torch.load('model_weights4434.pth')
    print("Starting training...")
    for k in range(epochs):
        start = time.time()
        xenc = F.one_hot(xs, num_classes=vocab_size).float()
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdims=True)
        loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W ** 2).mean()
        print(f'Epoch: {k + 1}, Loss: {loss.item()}')

        W.grad = None
        loss.backward()
        W.data += -learning_rate * W.grad
        tt =   time.time()-start
        ett = (time.time()-start)*(epochs-k)
        print("time :", tt, "Estimated: %s H %s m %s s"%(ett//3600,(ett%3600)//60,ett%60))
    print("Data trained!!")
    return W


def SampleData(test_data, W, seed, num_samples, max_len):
    itos = {i: word for word, i in vocab.items()}
    g = torch.Generator().manual_seed(seed)
    headlines = []
    i=0
    for headline in test_data:
        if i >= num_samples:
            break
        out = []
        ix = headline[0]
        print(itos[ix])
        while True:
            xenc = F.one_hot(torch.tensor([ix]), num_classes=4434).float()
            logits = xenc @ W
            counts = logits.exp()
            p = counts / counts.sum(1, keepdims=True)
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            if ix == 0:
                break
        if len(' '.join(out).split()) <= max_len+1:
            headlines.append(' '.join(out))
            i+=1
    return headlines

# # Prepare the data
token_ids, vocab, vocab_size = prepare_data('train1.csv')
#
# # Split the data into training and test sets
# # train_data, test_data = train_test_split_data(token_ids)
# train_data, test_data = token_ids, token_ids
# # Train the model
# W = TrainData(train_data, epochs=100, learning_rate=100)
#
# # Save the weights
# torch.save(W, "C:\\Users\\chaud\\Desktop\\Clickbait\\model_weights.pth")
#
# # Load the weights
# #W = torch.load('model_weights.pth')
#
# # Sample from the model
# SampleData(test_data, W, seed=42, num_samples=10)
