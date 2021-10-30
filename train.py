import os
import math
import argparse
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.nn import functional as F
from model import Encoder, Decoder, Seq2Seq, device
from utils import load_dataset


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=20,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=32,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='in case of gradient explosion')
    return p.parse_args()


def evaluate(model, val_iter, vocab_size, DE, EN, USER):
    model.eval()
    pad = EN.vocab.stoi['<pad>']
    eos_id = EN.vocab.stoi['<eos>']
    total_loss = 0
    decoded_batch_list = []
    with torch.no_grad():
        for b, batch in enumerate(val_iter):
            src, len_src = batch.src
            trg, len_trg = batch.trg
            user = batch.user
            src = Variable(src.data.to(device))
            trg = Variable(trg.data.to(device))
            user = Variable(user.data.to(device))
            output = model(src, trg, user, teacher_forcing_ratio=0.0)
            loss = F.nll_loss(output[1:].view(-1, vocab_size),
                              trg[1:].contiguous().view(-1),
                              ignore_index=pad)
            decoded_batch = model.decode(src, trg, user, method='beam-search')
            decoded_batch_list.append(decoded_batch)
            total_loss += loss.data.item()

        for sentence_index in decoded_batch_list[0]:
            decode_text_arr = [EN.vocab.itos[i] for i in sentence_index[0]]
            decode_stentence = " ".join(decode_text_arr[1:-1])
            print("pred target : {}".format(decode_stentence))

    return total_loss / (b + 1)


def train(e, model, optimizer, train_iter, vocab_size, grad_clip, DE, EN, USER):
    model.train()
    total_loss = 0
    pad = EN.vocab.stoi['<pad>']
    for b, batch in enumerate(train_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        user = batch.user
        src, trg, user = src.to(device), trg.to(device), user.to(device)
        optimizer.zero_grad()
        output = model(src, trg, user)
        loss = F.nll_loss(output[1:].view(-1, vocab_size),
                          trg[1:].contiguous().view(-1),
                          ignore_index=pad)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.data.item()

        if b % 100 == 0 and b != 0:
            total_loss = total_loss / 100
            print("[%d][loss:%5.2f][pp:%5.2f]" %
                  (b, total_loss, math.exp(total_loss)))
            total_loss = 0


def main():
    args = parse_arguments()
    hidden_size = 512
    embed_size = 256
    assert torch.cuda.is_available()

    print("[!] preparing dataset...")
    train_iter, val_iter, test_iter, DE, EN, USER = load_dataset(args.batch_size)
    de_size, en_size, user_size = len(DE.vocab), len(EN.vocab), len(USER.vocab)
    print("[TRAIN]:%d (dataset:%d)\t[TEST]:%d (dataset:%d)"
          % (len(train_iter), len(train_iter.dataset),
             len(test_iter), len(test_iter.dataset)))
    print("[DE_vocab]:%d [en_vocab]:%d [num_users]:%d" % (de_size, en_size, user_size))

    print("[!] Instantiating models...")
    encoder = Encoder(de_size, embed_size, hidden_size,
                      n_layers=2, dropout=0.5)
    decoder = Decoder(embed_size, hidden_size, en_size,
                      user_size,
                      n_layers=1, dropout=0.0)
    seq2seq = Seq2Seq(encoder, decoder).to(device)
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    print(seq2seq)

    best_val_loss = None
    for e in range(1, args.epochs + 1):
        train(e, seq2seq, optimizer, train_iter,
              en_size, args.grad_clip, DE, EN, USER)
        val_loss = evaluate(seq2seq, val_iter, en_size, DE, EN, USER)
        print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2f"
              % (e, val_loss, math.exp(val_loss)))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print("[!] saving model...")
            if not os.path.isdir(".save"):
                os.makedirs(".save")
            torch.save(seq2seq.state_dict(), './.save/seq2seq_%d.pt' % (e))
            best_val_loss = val_loss
    test_loss = evaluate(seq2seq, test_iter, en_size, DE, EN, USER)
    print("[TEST] loss:%5.2f" % test_loss)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
