#! /usr/bin/env python

from pathlib import Path
import argparse

import sentencepiece as spm

from dougu import lines


def argsparser():
    desc = "TODO"
    a = argparse.ArgumentParser(description=desc)
    a.add_argument("--input", type=Path, required=True)
    a.add_argument("--model", type=str, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--encode-as-pieces", action='store_true')
    return a


def main():
    args = argsparser().parse_args()
    s = spm.SentencePieceProcessor()
    s.Load(args.model)
    if args.encode_as_pieces:
        with args.output.open('w') as out:
            for line in lines(args.input):
                ids = s.EncodeAsPieces(line)
                out.write(' '.join(ids) + '\n')
    else:
        with args.output.open('w') as out:
            for line in lines(args.input):
                ids = s.EncodeAsIds(line)
                out.write(' '.join(map(str, ids)) + '\n')


if __name__ == "__main__":
    main()
