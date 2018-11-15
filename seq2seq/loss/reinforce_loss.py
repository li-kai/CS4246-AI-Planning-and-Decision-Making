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
        self.sampled_sentence = []
        self.greedy_sentence = []

    def get_loss(self):
        if isinstance(self.acc_loss, int):
            return 0
        # total loss for all batches
        loss = self.acc_loss
        # loss /= self.norm_term
        return loss

    def reset(self):
        self.sampled_sentence = []
        self.greedy_sentence = []

    def indices_to_words(self, inputs, keep_mask=False):
        result = []
        for x in inputs:
            if x != self.mask or keep_mask:
                result.append(self.itos(x))
        return result

    def matrix_bleu(self, matrix, targets, lengths):
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

    def teacher_forcing_eval_batch(self, outputs, greedy, sampled, lengths, target):
        batch_size, seq_length = target.size(0), len(outputs)
        acc_loss = torch.zeros((batch_size, 1)).cpu()

        for i in range(seq_length):
            acc_loss += torch.gather(outputs[i], 1, sampled[i])

        acc_loss = acc_loss.squeeze()
        self.acc_loss = self.acc_loss.mean()
        self.norm_term = batch_size

    def reinforce_eval_batch(self, outputs, greedy, sampled, lengths, target):
        # iter through time step
        # optimisation: do in matrix form
        batch_size, seq_length = target.size(0), len(outputs)
        acc_loss = torch.zeros((batch_size, 1)).cpu()

        target_sentences = [self.indices_to_words(sentence) for sentence in target]
        greedy_sentences = []
        sampled_sentences = []

        for i in range(seq_length):
            greedy_batch = self.indices_to_words(greedy[i], keep_mask=True)
            greedy_sentences.append([[x] for x in greedy_batch])
            sampled_batch = self.indices_to_words(sampled[i], keep_mask=True)
            sampled_sentences.append([[x] for x in sampled_batch])
            outputs[i] = outputs[i].cpu()
            sampled[i] = sampled[i].cpu()
            acc_loss += torch.gather(outputs[i], 1, sampled[i])

        greedy_sentences = np.concatenate(greedy_sentences, axis=1)
        sampled_sentences = np.concatenate(sampled_sentences, axis=1)
        # print(greedy_sentences[0])
        # print(sampled_sentences[0])

        sampled_bleu = self.matrix_bleu(sampled_sentences, target_sentences, lengths)
        greedy_bleu = self.matrix_bleu(greedy_sentences, target_sentences, lengths)
        acc_loss = acc_loss.squeeze()
        # print("==========================================")
        # print(sampled_bleu)
        # print(greedy_bleu)
        # print(acc_loss)
        # print((greedy_bleu - sampled_bleu))

        # (sample - greedy) should be positive (we want to go towards sampled)
        # loss should be positive, which we want to minimize
        # so we flip (sample - greedy) into (greedy - sample)
        self.acc_loss = (greedy_bleu - sampled_bleu) * acc_loss
        self.acc_loss = self.acc_loss.mean()
        # print(self.acc_loss)
        self.norm_term = batch_size

    def eval_batch(
        self, outputs, greedy, sampled, lengths, target, use_teacher_forcing
    ):
        if use_teacher_forcing:
            self.teacher_forcing_eval_batch(outputs, greedy, sampled, lengths, target)
        else:
            self.reinforce_eval_batch(outputs, greedy, sampled, lengths, target)
