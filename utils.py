import re
import spacy
from torchtext.data import Field, BucketIterator, TabularDataset


def load_dataset(batch_size):
    spacy_en = spacy.load('en')
    url = re.compile('(<url>.*</url>)')

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]

    sentence = Field(sequential=True, use_vocab=True, tokenize=tokenize_en, lower=True)

    # DE is input
    DE = Field(sequential=True, use_vocab=True, tokenize=tokenize_en, include_lengths=True, init_token='<sos>', eos_token='<eos>') 

    # EN is output
    EN = Field(sequential=True, use_vocab=True, tokenize=tokenize_en, include_lengths=True, init_token='<sos>', eos_token='<eos>') 

    USER = Field(sequential=False, use_vocab=True) 

    fields = {'user': ('user', USER), 'revision_text': ('trg', EN), 'parent_text': ('src', DE)}

    train, val, test = TabularDataset.splits(
                path='data',
                train='American_Philanthropists_norm_train.tsv',
                test='American_Philanthropists_norm_test.tsv',
                validation='American_Philanthropists_norm_val.tsv',
                format='tsv',
                fields=fields
            )

    DE.build_vocab(train.src, min_freq=2, max_size=10000)
    EN.build_vocab(train.trg, max_size=10000)
    USER.build_vocab(train.user, max_size=50000)
    train_iter, val_iter, test_iter = BucketIterator.splits(
            (train, val, test), batch_size=batch_size, repeat=False)
    return train_iter, val_iter, test_iter, DE, EN, USER

