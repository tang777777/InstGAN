import torch


class Tokenizer(object):

    def __init__(self, smiles):

        unique_char = list(set(''.join(smiles))) + ['<EOS>'] + ['<SOS>']

        self.mapping = {'<PAD>': 0}

        for i, c in enumerate(unique_char, start=1):
            self.mapping[c] = i

        self.inv_mapping = {v: k for k, v in self.mapping.items()}

        self.start_token = self.mapping['<SOS>']

        self.end_token = self.mapping['<EOS>']

        self.vocab_size = len(self.mapping.keys())

    def encode_smile(self, smile, eos_flag=True):

        encoded_smile = [self.mapping[i] for i in smile]

        if eos_flag:
            encoded_smile = encoded_smile + [self.end_token]

        return torch.LongTensor(encoded_smile)

    def batch_tokenize(self, batch):

        out = map(lambda x: self.encode_smile(x), batch)

        return torch.nn.utils.rnn.pad_sequence(list(out), batch_first=True)
