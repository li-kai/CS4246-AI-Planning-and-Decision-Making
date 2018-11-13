import random

source_file = "normal.aligned"
target_file = "simple.aligned"

source_train_file = "source_train.txt"
target_train_file = "target_train.txt"
source_test_file = "source_test.txt"
target_test_file = "target_test.txt"

line_pairs = []

with open(source_file, "r") as source, open(target_file, "r") as target:
    for src, tgt in zip(source, target):
        source_parsed = src.split('\t')[2]
        target_parsed = tgt.split('\t')[2]
        line_pairs.append((source_parsed, target_parsed))

test_lines = len(line_pairs) // 4
train_lines = len(line_pairs) - test_lines

train_line_pairs = random.sample(line_pairs, train_lines)
with open(source_train_file, "w") as source_train, open(target_train_file, "w") as target_train:
    for (src, tgt) in train_line_pairs:
        source_train.write(src)
        target_train.write(tgt)

test_line_pairs = random.sample(line_pairs, test_lines)
with open(source_test_file, "w") as source_test, open(target_test_file, "w") as target_test:
    for (src, tgt) in test_line_pairs:
        source_test.write(src)
        target_test.write(tgt)
