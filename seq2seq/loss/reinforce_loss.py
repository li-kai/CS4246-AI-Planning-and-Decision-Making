from nltk.translate.gleu_score import sentence_gleu
from .loss import NLLLoss

import math
import torch
import torch.nn as nn
import numpy as np


class BLEULoss(NLLLoss):
    def __init__(self, weight=None, mask=None, tgt=None):
        super(BLEULoss, self).__init__(
            weight=weight, mask=mask, reduction="none"
        )
        self.forcing_loss = NLLLoss(weight=weight, mask=mask, reduction="none")
        self.rl_loss = NLLLoss(weight=weight, mask=mask, reduction="none")
        self.tgt = tgt
        self.sampled_seq = []
        self.greedy_seq = []
        self.target_seq = []
        self.i = 0

    def reset(self):
        super(BLEULoss, self).reset()
        self.forcing_loss.reset()
        self.rl_loss.reset()
        self.sampled_seq = []
        self.greedy_seq = []
        self.target_seq = []

    def itos(self, sentence):
        output = []
        for x in sentence:
            if x == self.tgt.eos_id or x == self.mask:
                return output
            output.append(self.tgt.vocab.itos[x])
        return output

    def score_sentence(self, sentence, target):
        tgt = self.itos(target)
        sen = self.itos(sentence)
        # if (self.i == 0):
            # print(tgt)
            # print(sen)
            # print()
        self.i = 1

        return sentence_gleu([tgt], sen)

    def score(self, batch_of_sentences):
        self.i = 0
        scores = []
        for sentence, target in zip(batch_of_sentences, self.target_seq):
            scores.append(self.score_sentence(sentence, target))
        return torch.FloatTensor(scores).unsqueeze(1).cuda()

    '''
    Evaluates a training step, accumulating loss
    '''
    def eval_batch(self, outputs, target, sampled=None, greedy=None, use_teacher_forcing=True):
        # if use_teacher_forcing:
        #     # print("teacher", outputs.size(), target.size())
        #     super(BLEULoss, self).eval_batch(outputs, target)
        #     return
        self.forcing_loss.eval_batch(outputs, target)

        if sampled is None or greedy is None:
            raise ValueError("Missing sampled or greedy input")

        self.greedy_seq.append(greedy.squeeze())
        self.sampled_seq.append(sampled.squeeze())
        self.target_seq.append(target.squeeze())

        # accumulate log loss through parent class
        # print("non-teacher", outputs.size(), sampled.size())
        self.rl_loss.eval_batch(outputs, sampled.squeeze())

    def get_loss(self, use_teacher_forcing):
        self.greedy_seq = torch.stack(self.greedy_seq, dim=1)
        self.sampled_seq = torch.stack(self.sampled_seq, dim=1)
        self.target_seq = torch.stack(self.target_seq, dim=1)

        greedy_score = self.score(self.greedy_seq)
        sampled_score = self.score(self.sampled_seq)
        rl_loss = (sampled_score - greedy_score) * self.rl_loss.acc_loss
        forcing_loss = self.forcing_loss.acc_loss

        self.acc_loss = (0.5 * rl_loss + 0.5 * forcing_loss).mean()
        return self.acc_loss
