import torch
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.seq2seq_encoders import (LstmSeq2SeqEncoder,
                                               PytorchTransformer)
from torch import nn
from torch.distributions import Categorical
from torch.nn.modules.activation import Sigmoid


class Generator(nn.Module):

    def __init__(self, latent_dim, vocab_size, start_token, end_token):
        """Generator class:

        Args:
            latent_dim(int): the dimension for latent layers.
            vocab_size (int): the vocab size of training dataset (no padding).
            start_token (int): start token (no padding idx)
            end_token (int): end token (no padding idx)
        """

        super().__init__()

        # (-1) there is no padding token for generator
        self.vocab_size = vocab_size
        self.start_token = start_token
        self.end_token = end_token

        self.embedding_layer = nn.Embedding(self.vocab_size, latent_dim)

        self.project = FeedForward(
            input_dim=latent_dim,
            num_layers=2,
            hidden_dims=[latent_dim * 2, latent_dim * 2],
            activations=[nn.ReLU(), nn.ELU(alpha=0.1)],
            dropout=[0.1, 0.1]
        )

        self.rnn = nn.LSTMCell(latent_dim, latent_dim)

        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, vocab_size - 1)
        )

    def forward(self, noise, max_length=50):
        """
        Args:
            noise (torch.Tensor): input noise
            max_length (int): the max smile length.

        Return:
            dictionary types: x [batch_size, max_length], log_probabilities [batch_size, max_length, vocab size], entropies [batch size,]
        """

        batch_size = noise.shape[0]

        # sequences are start with start token
        starts = torch.full(
            size=(batch_size,), fill_value=self.start_token, device=noise.device).long()

        emb = self.embedding_layer(starts)

        x = []
        log_probabilities = []
        entropies = []

        h, c = self.project(noise).chunk(2, dim=1)

        for i in range(max_length):

            h, c = self.rnn(emb, (h, c))

            # logits
            logits = self.output_layer(h)

            # distribution
            dist = Categorical(logits=logits)

            # sampling
            sample = dist.sample()

            # append sampled atom
            x.append(sample)

            # append log probability
            log_probabilities.append(dist.log_prob(sample))

            # append entropy
            entropies.append(dist.entropy())

            # new embedding
            emb = self.embedding_layer(sample)

        # stack these returns along sequence dimension
        x = torch.stack(x, dim=1)
        log_probabilities = torch.stack(log_probabilities, dim=1)
        entropies = torch.stack(entropies, dim=1)

        # only uses the atoms before end token
        end_pos = (x == self.end_token).float().argmax(dim=1).cpu()

        # sequence length = the positon of end token + 1
        seq_lengths = end_pos + 1

        # if : end_pos = 0 > seq_length = max_len
        seq_lengths.masked_fill_(seq_lengths == 1, max_length)

        # select up to the length of each smiles
        _x = []
        _log_probabilities = []
        _entropies = []
        for x_i, logp, ent, length in zip(x, log_probabilities, entropies, seq_lengths):
            _x.append(x_i[:length])
            _log_probabilities.append(logp[:length])
            _entropies.append(ent[:length].mean())

        x = torch.nn.utils.rnn.pad_sequence(
            _x, batch_first=True, padding_value=-1)

        x = x + 1  # adding padding token for following processes

        return {'x': x, 'log_probabilities': _log_probabilities, 'entropies': _entropies}


class Discriminator(nn.Module):

    def __init__(self, latent_dim, vocab_size, start_token, bidirectional=True):
        """Discriminator class

        Args:
            latent_dim (int): model latent dimension size
            vocab_size (int): vocabulary size
            bidirectional (bool, optional): if the lstm is bidirectional 
        """

        super().__init__()

        self.start_token = start_token

        self.embedding = nn.Embedding(vocab_size, latent_dim, padding_idx=0)

        self.rnn = LstmSeq2SeqEncoder(
            latent_dim, latent_dim, num_layers=1, bidirectional=bidirectional)

        if bidirectional:
            latent_dim = latent_dim * 2

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, encoded_smiles):
        """[summary]

        Args:
            encoded_smiles : input encoded smiles

        Returns:
               prediction and mask
        """

        batch_size, _ = encoded_smiles.size()

        # input sequences are with start tokens
        starts = torch.full(
            size=(batch_size, 1), fill_value=self.start_token, device=encoded_smiles.device).long()

        encoded_smiles = torch.cat([starts, encoded_smiles], dim=1)

        mask = encoded_smiles > 0

        # embedding these input smiles [batch_size, seq_len, latent_dim_size]
        emb = self.embedding(encoded_smiles)

        # contextualize representation
        representations = self.rnn(emb, mask)

        # prediction for each token
        out = self.fc(representations).squeeze(-1)  # [batch_size, seq_len]

        return {'out': out[:, 1:], 'mask': mask.float()[:, 1:]}
