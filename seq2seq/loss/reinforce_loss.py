import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from .loss import NLLLoss


class BLEULoss(NLLLoss):
    """ Batch averaged (BLEU as reward * negative log-likelihood loss).

    Args:
        weight (torch.Tensor, optional): refer to http://pytorch.org/docs/master/nn.html#BLEULoss
        mask (int, optional): index of masked token, i.e. weight[mask] = 0.
        size_average (bool, optional): refer to http://pytorch.org/docs/master/nn.html#BLEULoss
    """

    _NAME = "Avg BLEULoss"

    def __init__(self, tgt_vocab, weight=None, mask=None):
        super(BLEULoss, self).__init__(weight=weight, mask=mask)
        self.tgt_vocab = tgt_vocab
        self.itos = np.vectorize(tgt_vocab.itos.__getitem__)
        self.sampled_sentence = []
        self.greedy_sentence = []

    def reset(self):
        self.sampled_sentence = []
        self.greedy_sentence = []

    def matrix_to_sentences(self, input):
        return self.itos(input)

    def eval_batch(self, outputs, greedy, sampled, lengths, target):
        # iter through time step
        # optimisation: do in matrix form
        batch_size, seq_length = target.size(0), len(outputs)
        acc_loss = torch.zeros((batch_size, seq_length))
        target_sentences = self.matrix_to_sentences(target)
        greedy_sentences = []
        sampled_sentences = []

        for i in range(len(outputs)):
            greedy_sentences.append(self.matrix_to_sentences(greedy[i]))
            sampled_sentences.append(self.matrix_to_sentences(sampled[i]))
            acc_loss += torch.gather(outputs[i], 1, sampled[i])

        greedy_sentences = np.concatenate(greedy_sentences, axis=1)
        sampled_sentences = np.concatenate(sampled_sentences, axis=1)

        print("------------------------------------------")
        print(greedy_sentences.shape)
        print(acc_loss.size())
        print(greedy_sentences)
        print("------------------------------------------")
        print(sampled_sentences)
        print("------------------------------------------")
        print(target_sentences)
        print("==========================================")
        greedy_bleu = sentence_bleu(greedy_sentences, target_sentences)
        sampled_bleu = sentence_bleu(sampled_sentences, target_sentences)
        print(sampled_bleu, greedy_bleu)
        self.acc_loss = (sampled_bleu - greedy_bleu) * acc_loss
        self.norm_term = batch_size
