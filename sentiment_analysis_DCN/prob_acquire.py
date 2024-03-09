import torch
import random
import numpy as np
from torch.backends import cudnn
from transformers import BertTokenizer

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

print(len(tokenizer.vocab))

tokens = tokenizer.tokenize('hello WORLD how ARE yoU?')
print(tokens)

indexs = tokenizer.convert_tokens_to_ids(tokens)
print(indexs)

init_token = tokenizer.cls_token
eos_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

print(init_token, eos_token, pad_token, unk_token)

init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

print(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx)

init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id

print(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx)

max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
print(max_input_length)


def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length - 2]
    return tokens


from torchtext import data

TEXT = data.Field(
    batch_first=True,
    use_vocab=False,
    tokenize=tokenize_and_cut,
    preprocessing=tokenizer.convert_tokens_to_ids,
    init_token=init_token_idx,
    eos_token=eos_token_idx,
    unk_token=unk_token_idx,
    pad_token=pad_token_idx
)

LABEL = data.LabelField(dtype=torch.float)

from torchtext import datasets

train_data, test_data = datasets.IMDB.splits(
    text_field=TEXT,
    label_field=LABEL
)

train_data, valid_data = train_data.split(random_state=random.seed(SEED))

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

print(vars(train_data.examples[6]))

tokens = tokenizer.convert_ids_to_tokens(vars(train_data.examples[6])['text'])
print(tokens)

LABEL.build_vocab(train_data)
print(LABEL.vocab.stoi)

BATCH_SIZE = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)

from transformers import BertTokenizer, BertModel

bert = BertModel.from_pretrained('bert-base-uncased')

import torch.nn as nn


class BERTGRUSentiment(nn.Module):
    def __init__(self, bert, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(BERTGRUSentiment, self).__init__()

        self.bert = bert

        embedding_dim = bert.config.to_dict()['hidden_size']

        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional,
                          batch_first=True, dropout=0 if n_layers < 2 else dropout)

        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):

        # text = [batch_size, sent_len]

        with torch.no_grad():
            embedded = self.bert(text)[0]

        # embedded = [batch_size, sent_len, emb_dim]

        _, hidden = self.rnn(embedded)

        # hidden = [n_layers * n_directions, batch_size, emb_dim]

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch_size, hidden_dim]

        output = self.out(hidden)

        # output = [batch_size, output_dim]

        return output


HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

model = BERTGRUSentiment(bert, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

for name, param in model.named_parameters():
    if name.startswith('bert'):
        param.requires_grad = False


def count_paraeters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

import torch.optim as optim

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()

            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


import time


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 5

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'Bert-model.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

model.load_state_dict(torch.load('Bert-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')


def predict_sentiment(model, tokenizer, sentence):
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length - 2]
    indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()


print("=" * 20)

s11 = "This is a great movie for a missionary going into a foreign country, especially one that is not used to foreign presence. But, it was a little on the short side."
s12 = 'This DVD appears to be in German. It is not in english. I do not speak German.  This needs to be corrected. I want my $1.99 back.'
s13 = 'This movie is not in English although the title of the movie is as is the book. There is no forewarning anywhere that the movie is not in English. Sounds like it might be Russian?'
s14 = 'This movie was in ENGLISH....it was a great summary of the book and the experience of the Richardsons while in New Guinea.'
s15 = 'More than anything, I\'ve been challenged to find ways to share Christ is a culturally relevant way to those around me.  Peace child is a cherished \"how to\" for me to do that.'

s21 = 'I have seen X live many times, both in the early days and their more recent reunion shows. Trust me when I say this: they never disappoint as a live band! This DVD document of a show on their home turf of LA is a dream come true. Can\'t wait.'
s22 = 'I think I saw X about 20 - 25 times from 1985 - 1995. They were always great and this brings back a lot of memories for me. They are a tight band that always sound great. I wish this had all of thier songs from that concert that is on the CD.'
s23 = 'I recently saw X live in Atlanta, Georgia and was blown away with thier amazing performance, this DVD displays how they still have a passion for playing punk rock music. Good picture and sound, if you like X, buy this testament of punk history!!!'
s24 = 'It is an excellent experience to watch this DVD after all these years of being an X fan.  It was also very cool to watch this right after a repeated viewing of the DVD X--The Unheard Music (1985).  X still has every ounce of energy and connection it had back then. The only reason I gave it four stars is that John Doe\'s Bass is overshadowed in the mix.  But having said that, it is a DVD I will treasure for years to come.'
s25 = 'This DVD is unbelievable!  If you are a fan of punk rock, you have to own this DVD.  They play all of their classic songs and the bonus footage is amazing.  I think my favorite part is when they play \"White Girl.\"  X has been around for a long time and this is their first live concert DVD, so it makes it even more awesome.  And the liner notes by Flea of The Red Hot Chili Peppers are really cool too.  Definitely buy this DVD!  It\'s worth every penny!'

s31 = 'I\'m happy with my purchase'
s32 = 'The little ones love this'
s33 = 'Love it!!  Great way to teach children the Bible in a fun way.  (Side bonus--adults are reminded of the Biblical truths again.)'
s34 = 'One of our family\'s favorites!  Still singing the songs 15 years later :)'
s35 = 'Good songs. The DVD is a little boring.'

s41 = 'Less than I expected, but OK filmc'
s42 = 'It is a great book about History!'
s43 = 'REALLY ENJOYED'
s44 = 'This hits our situation on the spot.'
s45 = 'If you love God and Country, this exposes another layer of the attack against both. well work the time watching ans sharing with your friends'

s51 = 'Nice movie'
s52 = 'A good movie with morals'
s53 = 'Love the twist on this classic!'
s54 = 'An inspirational version of the Christmas Carol, enjoyable and upbeat'
s55 = 'This TV movie brought back fond memories of the first time my wife and I watched it in the late 70\'s. Henry Winkler did a great job with the character of an American Scrooge. It was radical change from the Fonz.'

s61 = 'I was a movie I wanted in my collection for a very long time.  Saw it once and wanted to see it and own it.'
s62 = 'Classic 80\'s romance! :)'
s63 = 'eighties classic that now looks pretty silly.'
s64 = 'Best movie ever'
s65 = 'One of my favorite movies of all time!'

s71 = 'It\'s an adorable movies, great performances, highly enjoyable.'
s72 = 'best Cage movie ever and possibly best Conen brothers movie'
s73 = 'I was NEVER able to play this movie.  I hate that I had to pay for it.  Tried to speak to someone about it, but that did not get me a refund,'
s74 = 'Great movie.'
s75 = 'Hilarious!c'

s81 = 'Will be teaching this in a small Bible Study. Am exited to share what Ive read.'
s82 = 'My Bible study group loves the John\'s study!!  We are truly enjoying it!!'
s83 = 'Awesome spiritual guide'
s84 = 'Great study on a  topic that does not get enough attention.'
s85 = 'Excellent summary of a great work.'

s91 = 'Very interesting to watch!'
s92 = 'it is ok'
s93 = 'It left me with desiring more.  Wish it was longer and more thorough and more teaching.'
s94 = 'I felt as if I was there.'
s95 = 'There was way too much preaching in this video and not enough video of place where Jesus really did live and walk.  I was disappoined to say the least.'

s101 = 'Good to have on hand for those who don\'t know Jesus.'
s102 = 'I search for this version everywhere and I\'m so happy I found it. I grew up watching this version and although it\'s not the newest version the quality is amazing. I\'m buying a second one for my brother.'
s103 = 'Best depiction of the Gospel in my opinion.  I preder it over my many others since it is practically word for word from LUKE.'
s104 = 'Not what I expected.'
s105 = 'The most awe inspiring & powerful presentation of the Gospel of Luke you\'ll ever see.'

res11 = predict_sentiment(model, tokenizer, s11)
res12 = predict_sentiment(model, tokenizer, s12)
res13 = predict_sentiment(model, tokenizer, s13)
res14 = predict_sentiment(model, tokenizer, s14)
res15 = predict_sentiment(model, tokenizer, s15)
res21 = predict_sentiment(model, tokenizer, s21)
res22 = predict_sentiment(model, tokenizer, s22)
res23 = predict_sentiment(model, tokenizer, s23)
res24 = predict_sentiment(model, tokenizer, s24)
res25 = predict_sentiment(model, tokenizer, s25)
res31 = predict_sentiment(model, tokenizer, s31)
res32 = predict_sentiment(model, tokenizer, s32)
res33 = predict_sentiment(model, tokenizer, s33)
res34 = predict_sentiment(model, tokenizer, s34)
res35 = predict_sentiment(model, tokenizer, s35)
res41 = predict_sentiment(model, tokenizer, s41)
res42 = predict_sentiment(model, tokenizer, s42)
res43 = predict_sentiment(model, tokenizer, s43)
res44 = predict_sentiment(model, tokenizer, s44)
res45 = predict_sentiment(model, tokenizer, s45)
res51 = predict_sentiment(model, tokenizer, s51)
res52 = predict_sentiment(model, tokenizer, s52)
res53 = predict_sentiment(model, tokenizer, s53)
res54 = predict_sentiment(model, tokenizer, s54)
res55 = predict_sentiment(model, tokenizer, s55)
res61 = predict_sentiment(model, tokenizer, s61)
res62 = predict_sentiment(model, tokenizer, s62)
res63 = predict_sentiment(model, tokenizer, s63)
res64 = predict_sentiment(model, tokenizer, s64)
res65 = predict_sentiment(model, tokenizer, s65)
res71 = predict_sentiment(model, tokenizer, s71)
res72 = predict_sentiment(model, tokenizer, s72)
res73 = predict_sentiment(model, tokenizer, s73)
res74 = predict_sentiment(model, tokenizer, s74)
res75 = predict_sentiment(model, tokenizer, s75)
res81 = predict_sentiment(model, tokenizer, s81)
res82 = predict_sentiment(model, tokenizer, s82)
res83 = predict_sentiment(model, tokenizer, s83)
res84 = predict_sentiment(model, tokenizer, s84)
res85 = predict_sentiment(model, tokenizer, s85)
res91 = predict_sentiment(model, tokenizer, s91)
res92 = predict_sentiment(model, tokenizer, s92)
res93 = predict_sentiment(model, tokenizer, s93)
res94 = predict_sentiment(model, tokenizer, s94)
res95 = predict_sentiment(model, tokenizer, s95)
res101 = predict_sentiment(model, tokenizer, s101)
res102 = predict_sentiment(model, tokenizer, s102)
res103 = predict_sentiment(model, tokenizer, s103)
res104 = predict_sentiment(model, tokenizer, s104)
res105 = predict_sentiment(model, tokenizer, s105)

print("=" * 10, 1, "=" * 10)
print(res11)
print(res12)
print(res13)
print(res14)
print(res15)

print("=" * 10, 2, "=" * 10)
print(res21)
print(res22)
print(res23)
print(res24)
print(res25)

print("=" * 10, 3, "=" * 10)
print(res31)
print(res32)
print(res33)
print(res34)
print(res35)

print("=" * 10, 4, "=" * 10)
print(res41)
print(res42)
print(res43)
print(res44)
print(res45)

print("=" * 10, 5, "=" * 10)
print(res51)
print(res52)
print(res53)
print(res54)
print(res55)

print("=" * 10, 6, "=" * 10)
print(res61)
print(res62)
print(res63)
print(res64)
print(res65)

print("=" * 10, 7, "=" * 10)
print(res71)
print(res72)
print(res73)
print(res74)
print(res75)

print("=" * 10, 8, "=" * 10)
print(res81)
print(res82)
print(res83)
print(res84)
print(res85)

print("=" * 10, 9, "=" * 10)
print(res91)
print(res92)
print(res93)
print(res94)
print(res95)

print("=" * 10, 10, "=" * 10)
print(res101)
print(res102)
print(res103)
print(res104)
print(res105)
