import os
import argparse
import logging

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import NLLLoss
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor, Evaluator
from seq2seq.util.checkpoint import Checkpoint

# Path to experiment directory for storing checkpoints
EXPERIMENT_PATH = "./experiment"

# Sample usage:
#     # training
#      TRAIN_PATH=data/simple_wiki/train/data.txt
#      TEST_PATH=data/simple_wiki/test/data.txt
#      python seq2seq.py --train_path $TRAIN_PATH --test_path $TEST_PATH
#     # resuming from the latest checkpoint of the experiment
#      python seq2seq.py --train_path $TRAIN_PATH --test_path $TEST_PATH --resume
#     # resuming from a specific checkpoint
#      python seq2seq.py --train_path $TRAIN_PATH --test_path $TEST_PATH --load_checkpoint $CHECKPOINT_DIR

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_path",
    action="store",
    dest="train_path",
    help="Path to train data",
    default="data/simple_wiki/train/data.txt",
)
parser.add_argument(
    "--test_path",
    action="store",
    dest="test_path",
    help="Path to test data",
    default="data/simple_wiki/test/data.txt",
)
parser.add_argument(
    "--load_checkpoint",
    action="store",
    dest="load_checkpoint",
    help="The name of the checkpoint to load, usually an encoded time string",
)
parser.add_argument(
    "--resume",
    action="store_true",
    dest="resume",
    default=False,
    help="Indicates if training has to be resumed from the latest checkpoint",
)
parser.add_argument(
    "--log-level", dest="log_level", default="info", help="Logging level."
)

opt = parser.parse_args()

LOG_FORMAT = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

if opt.load_checkpoint is not None:
    checkpoint_path = os.path.join(
        EXPERIMENT_PATH, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint
    )
    logging.info("loading checkpoint from {}".format(checkpoint_path))
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab

    src = SourceField(sequential=True, use_vocab=True)
    tgt = TargetField(sequential=True, use_vocab=True)
    src.vocab = input_vocab
    tgt.vocab = output_vocab

    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = NLLLoss(weight, pad)
else:
    # Prepare dataset
    src = SourceField(sequential=True, use_vocab=True)
    tgt = TargetField(sequential=True, use_vocab=True)
    max_len = 25 

    train = torchtext.data.TabularDataset(
        path=opt.train_path, format="tsv", fields=[("src", src), ("tgt", tgt)]
    )
    src.build_vocab(train, vectors="glove.6B.100d", max_size=16384)
    tgt.build_vocab(train, vectors="glove.6B.100d", max_size=16384)
    input_vocab = src.vocab
    output_vocab = tgt.vocab

    # NOTE: If the source field name and the target field name
    # are different from 'src' and 'tgt' respectively, they have
    # to be set explicitly before any training or inference
    # seq2seq.src_field_name = 'src'
    # seq2seq.tgt_field_name = 'tgt'

    # Prepare loss
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = NLLLoss(weight, pad)

    seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        hidden_size = 100
        bidirectional = True 

        encoder = EncoderRNN(
            len(src.vocab),
            max_len,
            hidden_size,
            embedding=src.vocab.vectors,
            bidirectional=bidirectional,
            variable_lengths=True,
            rnn_cell="lstm",
        )
        decoder = DecoderRNN(
            len(tgt.vocab),
            max_len,
            hidden_size * 2 if bidirectional else hidden_size,
            # dropout_p=0.2,
            use_attention=True,
            bidirectional=bidirectional,
            eos_id=tgt.eos_id,
            sos_id=tgt.sos_id,
            rnn_cell="lstm",
        )
        seq2seq = Seq2seq(encoder, decoder)
        if torch.cuda.is_available():
            seq2seq.cuda()

        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

        # Optimizer and learning rate scheduler can be customized by
        # explicitly constructing the objects and pass to the trainer.
        #
        # optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
        # scheduler = StepLR(optimizer.optimizer, 1)
        # optimizer.set_scheduler(scheduler)

    # train
    t = SupervisedTrainer(
        loss=loss,
        batch_size=64,
        checkpoint_every=50,
        print_every=10,
        expt_dir=EXPERIMENT_PATH,
    )

    seq2seq = t.train(
        seq2seq,
        train,
        num_epochs=6,
        optimizer=optimizer,
        teacher_forcing_ratio=0.5,
        resume=opt.resume,
    )

predictor = Predictor(seq2seq, input_vocab, output_vocab)
loss, acc = Evaluator(loss=loss).evaluate(
    seq2seq,
    torchtext.data.TabularDataset(
        path=opt.test_path, format="tsv", fields=[("src", src), ("tgt", tgt)]
    ),
)
logging.info("Loss: {}, Acc: {}".format(loss, acc))

while True:
    seq_str = input("Type in a source sequence:")
    seq = seq_str.strip().split()
    print(predictor.predict(seq))
