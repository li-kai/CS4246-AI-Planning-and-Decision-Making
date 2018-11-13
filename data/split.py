import random
from nltk.tokenize.nist import NISTTokenizer

MIN_LEN = 10
MAX_LEN = 30

source_file = "normal.aligned"
target_file = "simple.aligned"

source_train_file = "source_train.txt"
target_train_file = "target_train.txt"
source_test_file = "source_test.txt"
target_test_file = "target_test.txt"

line_pairs = []

nist = NISTTokenizer()

with open(source_file, "r") as source, open(target_file, "r") as target:
    for src, tgt in zip(source, target):
        target_parsed = tgt.split('\t')[2]
        if not(MIN_LEN < len(target_parsed.split()) < MAX_LEN):
            continue
        source_parsed = src.split('\t')[2]
        if not(MIN_LEN < len(source_parsed.split()) < MAX_LEN):
            continue

        target_tokenised = nist.tokenize(target_parsed, lowercase=True)
        if not(MIN_LEN < len(target_tokenised.split()) < MAX_LEN):
            continue
        source_tokenised = nist.tokenize(source_parsed, lowercase=True)
        if not(MIN_LEN < len(source_tokenised.split()) < MAX_LEN):
            continue

        source_joined = " ".join(source_tokenised) + "\n"
        target_joined = " ".join(target_tokenised) + "\n"
        line_pairs.append((source_joined, target_joined))

test_lines = len(line_pairs) // 4
train_lines = len(line_pairs) - test_lines

print("Lines in train:", train_lines)
print("Lines in test:", test_lines)

train_line_pairs = random.sample(line_pairs, train_lines)
with open(source_train_file, "w") as source_train, open(target_train_file, "w") as target_train:
    src, tgt = zip(*train_line_pairs)
    source_train.writelines(src)
    target_train.writelines(tgt)

test_line_pairs = random.sample(line_pairs, test_lines)
with open(source_test_file, "w") as source_test, open(target_test_file, "w") as target_test:
    src, tgt = zip(*test_line_pairs)
    source_test.writelines(src)
    target_test.writelines(tgt)
