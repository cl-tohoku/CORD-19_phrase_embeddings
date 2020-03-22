#! /usr/bin/env python
import sentencepiece as spm
import sys


def main():
    spm.SentencePieceTrainer.Train(' '.join(sys.argv[1:]))


if __name__ == "__main__":
    main()
