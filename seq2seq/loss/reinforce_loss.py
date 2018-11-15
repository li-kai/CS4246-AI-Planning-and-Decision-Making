from nltk.translate.bleu_score import corpus_bleu
from .loss import NLLLoss


class BLEUoss(NLLLoss):
    """ Batch averaged (BLEU as reward * negative log-likelihood loss).

    Args:
        weight (torch.Tensor, optional): refer to http://pytorch.org/docs/master/nn.html#BLEUoss
        mask (int, optional): index of masked token, i.e. weight[mask] = 0.
        size_average (bool, optional): refer to http://pytorch.org/docs/master/nn.html#BLEUoss
    """

    _NAME = "Avg BLEUoss"

    def __init__(self, weight=None, mask=None):
        super(BLEUoss, self).__init__(weight=weight, mask=mask)

    def eval_batch(self, outputs, sampled_outputs, target):
        # TODO: pass in sampled output
        # TODO: turn weight into words
        print(outputs)
        print(sampled_outputs)
        # sampled_bleu = corpus_bleu(outputs, target)
        # greedy_bleu = corpus_bleu(outputs, target)
        acc_loss = self.criterion(outputs, target)
        # acc_loss *= (sampled_bleu - greedy_bleu)
        self.acc_loss += acc_loss
        self.norm_term += 1
