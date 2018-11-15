import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
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
        self.itos = tgt_vocab.itos.__getitem__
        self.smoothing = SmoothingFunction()

    def get_loss(self):
        if isinstance(self.acc_loss, int):
            return 0
        # total loss for all batches
        loss = self.acc_loss
        # loss /= self.norm_term
        return loss

    def indices_to_words(self, inputs, keep_mask=False):
        if keep_mask:
            return [self.itos(x) for x in inputs]
        else:
            return [self.itos(x) for x in inputs if x != self.mask]

    def matrix_bleu(self, matrix, targets):
        scores = []
        for i in range(len(matrix)):
            source_sentence = matrix[i]
            target_sentence = targets[i]
            x = sentence_bleu(
                [target_sentence],
                source_sentence,
                smoothing_function=self.smoothing.method1,
            )
            scores.append(x)
        # print("->", scores)
        return torch.FloatTensor(scores)

    def eval_batch(self, outputs, greedy, sampled, lengths, target):
        # iter through time step
        # optimisation: do in matrix form
        batch_size, seq_length = target.size(0), len(outputs)
        acc_loss = torch.zeros(batch_size).cpu()

        target_sentences = [self.indices_to_words(sentence) for sentence in target]

        zeros = torch.zeros(batch_size, requires_grad=False)
        for i in range(seq_length):
            outputs[i] = outputs[i].cpu()
            greedy[i] = greedy[i].cpu()
            sampled[i] = sampled[i].cpu()

            probs = torch.gather(outputs[i], 1, sampled[i]).squeeze()
            mask = sampled[i].squeeze().ne(self.mask)
            # zero out where sample index is mask
            loss = torch.where(mask, probs, zeros)
            acc_loss += loss

        greedy_indices = np.concatenate(greedy, axis=1)
        sampled_indices = np.concatenate(sampled, axis=1)

        greedy_sentences = []
        sampled_sentences = []
        for i in range(batch_size):
            length = lengths[i]
            sentence = self.indices_to_words(greedy_indices[i, :length], keep_mask=True)
            greedy_sentences.append(sentence)
            sentence = self.indices_to_words(sampled_indices[i])
            sampled_sentences.append(sentence)
        print(" ".join(greedy_sentences[0]))
        print(" ".join(sampled_sentences[0]))
        print(" ".join(target_sentences[0]))
        print("==========================================")

        sampled_bleu = self.matrix_bleu(sampled_sentences, target_sentences)
        greedy_bleu = self.matrix_bleu(greedy_sentences, target_sentences)
        # print(acc_loss)
        # print((greedy_bleu - sampled_bleu))

        # (sample - greedy) should be positive (we want to go towards sampled)
        # loss should be positive, which we want to minimize
        # so we flip (sample - greedy) into (greedy - sample)
        self.acc_loss = (greedy_bleu - sampled_bleu) * acc_loss
        self.acc_loss = self.acc_loss.mean()
        # print(self.acc_loss)
        self.norm_term = batch_size
