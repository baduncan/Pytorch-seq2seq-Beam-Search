import math
import operator
import random
from queue import PriorityQueue

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from log import timeit

SOS_token = 2
EOS_token = 3
MAX_LENGTH = 50

device = torch.device('cuda:2')

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size,
                 ed_size,
                 ed_embed_size,
                 n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size  # 8014
        self.hidden_size = hidden_size  # 512
        self.embed_size = embed_size  # 256
        self.ed_size = ed_size
        self.ed_embed_size = ed_embed_size
        self.ed_embed = nn.Embedding(ed_size, ed_embed_size)
        self.embed = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size + ed_embed_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, src, ed, hidden=None):
        embedded = self.embed(src)  # [max_len, batch_size]
        ed_embedded = self.ed_embed(ed)
        full_embedded = torch.cat([embedded, ed_embedded], 2)
        outputs, hidden = self.gru(full_embedded, hidden)  # ([27, 32, 256],None)=>([27, 32, 1024],[4, 32, 512])
        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])  # =>[27, 32, 512] + [27, 32, 512]
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)  # [32, 512]=>[32, 27, 512]
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H] # [27, 32, 512]=>[32,27,512]
        attn_energies = self.score(h, encoder_outputs)  # =>[B*T]
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # [B*T]=>[B*1*T]

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*H] bmm [B*H*T]=>[B*1*T]
        return energy.squeeze(1)  # [B*T]


class Decoder_noUsers(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 user_size, # ignored
                 n_layers=1, dropout=0.2):
        super(Decoder_noUsers, self).__init__()
        self.embed_size = embed_size  # 256
        self.hidden_size = hidden_size  # 512
        self.output_size = output_size  # 10004
        self.n_layers = n_layers  # 1

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size,
                          n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, user, encoder_outputs):  # output, hidden_state
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N) # [32]=>[32, 256]=>[1, 32, 256]
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)  # [32, 512][27, 32, 512]=>[32, 1, 27]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N) # [32, 1, 27]bmm[32, 27, 512]=>[32,1,512]
        context = context.transpose(0, 1)  # (1,B,N) # [32, 1, 512]=>[1, 32, 512]
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)  # [1, 32, 256] cat [1, 32, 512]=> [1, 32, 768]
        output, hidden = self.gru(rnn_input, last_hidden)  # in:[1, 32, 768],[1, 32, 512]=>[1, 32, 512],[1, 32, 512]
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))  # [32, 512] cat [32, 512] => [32, 512*2]
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights  # [32, 10004] [1, 32, 512] [32, 1, 27]


class Decoder_wUsers(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size,
                 user_size,
                 n_layers=1, dropout=0.2):
        super(Decoder_wUsers, self).__init__()
        self.embed_size = embed_size  # 256
        self.hidden_size = hidden_size  # 512
        self.output_size = output_size  # 10004
        self.n_layers = n_layers  # 1

        self.user_size = user_size
        self.user_embed = nn.Embedding(user_size, embed_size)

        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size + embed_size, hidden_size,
                          n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, user, encoder_outputs):  # output, hidden_state

        embedded_user = self.user_embed(user).unsqueeze(0)

        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # (1,B,N) # [32]=>[32, 256]=>[1, 32, 256]
        embedded = self.dropout(embedded)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)  # [32, 512][27, 32, 512]=>[32, 1, 27]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N) # [32, 1, 27]bmm[32, 27, 512]=>[32,1,512]
        context = context.transpose(0, 1)  # (1,B,N) # [32, 1, 512]=>[1, 32, 512]

        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context, embedded_user], 2)  # [1, 32, 256] cat [1, 32, 512]=> [1, 32, 768]
        output, hidden = self.gru(rnn_input, last_hidden)  # in:[1, 32, 768],[1, 32, 512]=>[1, 32, 512],[1, 32, 512]
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))  # [32, 512] cat [32, 512] => [32, 512*2]
        output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights  # [32, 10004] [1, 32, 512] [32, 1, 27]


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, user, ed, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).to(device)

        encoder_output, hidden = self.encoder(src, ed)  # [27, 32]=> =>[27, 32, 512],[4, 32, 512]
        hidden = hidden[:self.decoder.n_layers]  # [4, 32, 512][1, 32, 512]
        output = Variable(trg.data[0, :])  # sos
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(
                output, hidden, user, encoder_output)  # output:[32, 10004] [1, 32, 512] [32, 1, 27]
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]  # dim=1 ,x[1] =>top1.shape=32
            output = Variable(trg.data[t] if is_teacher else top1).to(device)
        return outputs

    def decode(self, src, trg, user, ed, method='beam-search'):
        encoder_output, hidden = self.encoder(src, ed)  # [27, 32]=> =>[27, 32, 512],[4, 32, 512]
        hidden = hidden[:self.decoder.n_layers]  # [4, 32, 512][1, 32, 512]
        if method == 'beam-search':
            return self.beam_decode(trg, hidden, user, encoder_output)
        else:
            return self.greedy_decode(trg, hidden, user, encoder_output)

    def greedy_decode(self, trg, decoder_hidden, user, encoder_outputs, ):
        '''
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :return: decoded_batch
        '''
        seq_len, batch_size = trg.size()
        decoded_batch = torch.zeros((batch_size, seq_len))
        # decoder_input = torch.LongTensor([[EN.vocab.stoi['<sos>']] for _ in range(batch_size)]).to(device)
        decoder_input = Variable(trg.data[0, :]).to(device)  # sos
        for t in range(seq_len):
            decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, user, encoder_outputs)

            topv, topi = decoder_output.data.topk(1)  # [32, 10004] get candidates
            topi = topi.view(-1)
            decoded_batch[:, t] = topi

            decoder_input = topi.detach().view(-1)

        return decoded_batch

    @timeit
    def beam_decode(self, target_tensor, decoder_hiddens, user, encoder_outputs=None):
        '''
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder_hiddens: input tensor of shape [1, B, H] for start of the decoding
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :return: decoded_batch
        '''
        target_tensor = target_tensor.permute(1, 0)
        beam_width = 10
        topk = 1  # how many sentence do you want to generate
        decoded_batch = []

        # decoding goes sentence by sentence
        for idx in range(target_tensor.size(0)):  # batch_size
            if isinstance(decoder_hiddens, tuple):  # LSTM case
                decoder_hidden = (
                    decoder_hiddens[0][:, idx, :].unsqueeze(0), decoder_hiddens[1][:, idx, :].unsqueeze(0))
            else:
                decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)  # [1, B, H]=>[1,H]=>[1,1,H]
            decoder_user = user[idx].unsqueeze(0) # BAD added
            encoder_output = encoder_outputs[:, idx, :].unsqueeze(1)  # [T,B,H]=>[T,H]=>[T,1,H]

            # Start with the start of the sentence token
            decoder_input = torch.LongTensor([SOS_token]).to(device)

            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1

            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 2000: break

                # fetch the best node
                score, n = nodes.get()
                # print('--best node seqs len {} '.format(n.leng))
                decoder_input = n.wordid
                decoder_hidden = n.h

                if n.wordid.item() == EOS_token and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                # call to decoder.forward(self, input, last_hidden, user, encoder_outputs):  # output, hidden_state
                decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, decoder_user, encoder_output)

                # PUT HERE REAL BEAM SEARCH OF TOP
                log_prob, indexes = torch.topk(decoder_output, beam_width)
                nextnodes = []

                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].view(-1)
                    log_p = log_prob[0][new_k].item()

                    node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid)
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid)

                utterance = utterance[::-1]
                utterances.append(utterance)

            decoded_batch.append(utterances)

        return decoded_batch


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

    def __lt__(self, other):
        return self.leng < other.leng

    def __gt__(self, other):
        return self.leng > other.leng
