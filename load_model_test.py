import sys
import argparse
from model import Encoder, Decoder_noUsers, Decoder_wUsers, Seq2Seq, device
from utils import load_dataset
import torch
from torch.autograd import Variable
from torch.nn import functional as F

import numpy as np
from nltk.translate.bleu_score import sentence_bleu

def evaluate(model, val_iter, vocab_size, DE, EN, USER, output_path):
    model.eval()
    pad = EN.vocab.stoi['<pad>']
    eos_id = EN.vocab.stoi['<eos>']
    total_loss = 0
    src_batch_list = []
    trg_batch_list = []
    decoded_batch_list = []
    lid_batch_list = []
    with torch.no_grad():
        for b, batch in enumerate(val_iter):
            src, len_src = batch.src
            trg, len_trg = batch.trg
            user = batch.user
            lid = batch.lid
            src = Variable(src.data.to(device))
            trg = Variable(trg.data.to(device))
            user = Variable(user.data.to(device))
            lid = Variable(lid.data.to(device))

            ed, len_ed = batch.ed
            ed = Variable(ed.data.to(device))

            output = model(src, trg, user, ed, teacher_forcing_ratio=0.0)
            loss = F.nll_loss(output[1:].view(-1, vocab_size),
                              trg[1:].contiguous().view(-1),
                              ignore_index=pad)
            decoded_batch = model.decode(src, trg, user, ed, method='beam-search')
            decoded_batch_list.append(decoded_batch)
            lid_batch_list.append(lid)
            total_loss += loss.data.item()

        with open(output_path, 'w') as outfile:
                    
            for blidx in range( len(decoded_batch_list) ):
                lid_for_batch = lid_batch_list[blidx]
                print(lid_for_batch)
                for idx, sentence_index in enumerate( decoded_batch_list[blidx] ):
                    decode_text_arr = [EN.vocab.itos[i] for i in sentence_index[0]]
                    decode_sentence = " ".join(decode_text_arr[1:-1])
                    outfile.write(decode_sentence + '\t')

                    lid = int(lid_for_batch[idx])
                    outfile.write(str(lid) + '\n')

    return total_loss / (b + 1)

def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-noUsers', type=bool, default=False,
                   help='True if running without user embeddings')
    p.add_argument('-model_directory', type=str, default=10.0,
                   help='Path to model file to load, must match -noUsers type')
    p.add_argument('-batch_size', type=int, default=32,
                   help='number of epochs for train')
    return p.parse_args()

def main():
    args = parse_arguments()
    model_path = args.model_directory

    output_val_path = '/'.join( model_path.split('/')[:-1] ) + '/val_out.tsv'
    output_test_path = '/'.join( model_path.split('/')[:-1] ) + '/test_out.tsv'

    hidden_size = 512
    embed_size = 256
    ed_embed_size = 32
    assert torch.cuda.is_available()

    print("[!] preparing dataset...")
    train_iter, val_iter, test_iter, DE, EN, USER, ED = load_dataset(args.batch_size)
    de_size, en_size, user_size, ed_size = len(DE.vocab), len(EN.vocab), len(USER.vocab), len(ED.vocab)

    print("[TRAIN]:%d (dataset:%d)\t[TEST]:%d (dataset:%d)"
          % (len(train_iter), len(train_iter.dataset),
             len(test_iter), len(test_iter.dataset)))
    print("[DE_vocab]:%d [en_vocab]:%d [num_users]:%d [num_edit_types]:%d" % (de_size, en_size, user_size, ed_size))

    print("[!] Loading model...")
    encoder = Encoder(de_size, embed_size, hidden_size,
                      ed_size,
                      ed_embed_size,
                      n_layers=2, dropout=0.5)

    if args.noUsers:
        decoder = Decoder_noUsers(embed_size, hidden_size, en_size,
                      user_size,
                      n_layers=1, dropout=0.0)
    else:
        decoder = Decoder_wUsers(embed_size, hidden_size, en_size,
                      user_size,
                      n_layers=1, dropout=0.0)

    seq2seq = Seq2Seq(encoder, decoder).to(device)

    seq2seq.load_state_dict(torch.load(model_path))

    val_loss = evaluate(seq2seq, val_iter, en_size, DE, EN, USER, output_val_path)
    test_loss = evaluate(seq2seq, test_iter, en_size, DE, EN, USER, output_test_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
