import random
import spacy
import os

nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner"])

MIN_LEN = 3
MAX_LEN = 25

source_file = "normal.aligned"
target_file = "simple.aligned"

if not os.path.exists("train"):
    os.mkdir("train")
if not os.path.exists("test"):
    os.mkdir("test")
train_file = os.path.join("train", "data.txt")
test_file = os.path.join("test", "data.txt")

line_pairs = []

def is_unicode(mystring):
    try:
        mystring.encode('ascii')
    except UnicodeEncodeError:
        return True
    else:
        return False

with open(source_file, "r") as source, open(target_file, "r") as target:
    for src, tgt in zip(source, target):
        target_parsed = tgt.split("\t")[2].rstrip()
        if is_unicode(target_parsed) or not (MIN_LEN < target_parsed.count(" ") < MAX_LEN):
            continue
        source_parsed = src.split("\t")[2].rstrip()
        if is_unicode(target_parsed) or not (MIN_LEN < source_parsed.count(" ") < MAX_LEN):
            continue

        target_tokenised = [w.text for w in nlp(target_parsed.lower())]
        if not (MIN_LEN < len(target_tokenised) < MAX_LEN):
            continue
        source_tokenised = [w.text for w in nlp(source_parsed.lower())]
        if not (MIN_LEN < len(source_tokenised) < MAX_LEN):
            continue

        source_joined = " ".join(source_tokenised)
        target_joined = " ".join(target_tokenised)
        line_pairs.append((source_joined, target_joined))

random.shuffle(line_pairs)
split_point = len(line_pairs) - (len(line_pairs) // 4)
print("Train lines:", split_point, "Test lines:", len(line_pairs) // 4)
train_line_pairs = line_pairs[:split_point]
test_line_pairs = line_pairs[split_point:]

with open(train_file, "w") as train:
    train.writelines([src + "\t" + tgt + "\n" for src, tgt in train_line_pairs])

with open(test_file, "w") as test:
    test.writelines([src + "\t" + tgt + "\n" for src, tgt in train_line_pairs])
