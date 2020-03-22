#! /usr/bin/env python

from pathlib import Path
import argparse
import logging

from dougu import mkdir


logging.basicConfig(level=logging.WARNING)


def argsparser():
    desc = "TODO"
    a = argparse.ArgumentParser(description=desc)
    a.add_argument("--spm-id-emb", type=Path, required=True)
    a.add_argument("--vocab-file", type=Path, required=True)
    a.add_argument("--vocab-size", type=int, required=True)
    a.add_argument("--outdir", type=Path, default='emb')
    a.add_argument("--overwrite", action="store_true")
    a.add_argument("--check-corrupt", action="store_true")
    a.add_argument("--delete-corrupt", action="store_true")
    return a


def get_vocab(vocab_file, vocab_size):
    with vocab_file.open(encoding="utf8") as f:
        # read lines, ignoring fun characters such as 'LINE SEPARATOR' (U+2028)
        # which Python treats as line breaks when reading files
        # with the ususal 'for line in f' pattern
        vocab_lines = f.read().split("\n")[:-1]
    assert len(vocab_lines) == vocab_size
    vocab, ranks = zip(*map(lambda l: l.split("\t"), vocab_lines))
    return vocab


def check_corrupt(emb_file, vocab_size=None, delete_corrupt=False):
    from dougu import load_word2vec_file
    import numpy as np
    try:
        emb = load_word2vec_file(emb_file)
        assert not (emb.vectors == np.inf).any()
        assert not np.isnan(emb.vectors).any()
        if vocab_size is not None:
            assert len(emb.vocab) == vocab_size
            assert emb.vectors.shape[0] == vocab_size
        return emb
    except (ValueError, AssertionError, EOFError):
        if delete_corrupt:
            emb_file.unlink()
            print("deleted", emb_file)
        else:
            print(emb_file)
            import traceback
            traceback.print_exc()


def convert_emb(spm_id_emb, args):
    fname1 = spm_id_emb.stem
    fname2 = spm_id_emb.with_suffix('.bin').name
    outdir = args.outdir
    outfile1 = outdir / fname1
    outfile2 = outdir / fname2
    id_emb = None
    if args.check_corrupt:
        id_emb = check_corrupt(
            spm_id_emb, delete_corrupt=args.delete_corrupt)
        if id_emb is None:
            return
    for outfile in outfile1, outfile2:
        if args.check_corrupt:
            if outfile.exists():
                check_corrupt(outfile, args.vocab_size, args.delete_corrupt)

    if outfile1.exists() and outfile2.exists() and not args.overwrite:
        # print(outfile1)
        # print(outfile2)
        return

    import numpy as np
    from gensim.models import keyedvectors
    from dougu import load_word2vec_file, to_from_idx

    vocab = get_vocab(args.vocab_file, args.vocab_size)
    piece2id, id2piece = to_from_idx(vocab)
    if id_emb is None:
        id_emb = load_word2vec_file(spm_id_emb)
    v = id_emb.vectors
    # sample embeddings for symbols that didn't occur in the training
    # data from normal distribution with same mean and variance
    new_v = v.std() * np.random.randn(len(vocab), v.shape[1]) + v.mean()
    new_vocab = {}
    for id, piece in id2piece.items():
        try:
            new_v[id] = id_emb[str(id)]
        except KeyError:
            pass
        # gensim sorts embeddings by -count when saving
        # set count to -id to preserve sentencepiece order
        assert piece not in new_vocab
        new_vocab[piece] = keyedvectors.Vocab(count=-id, index=id)

    id_emb.index2word = id2piece
    id_emb.vocab = new_vocab
    id_emb.vectors = new_v
    bpemb = id_emb

    for ext, binary in zip((".txt", ".bin"), (False, True)):
        fname = spm_id_emb.with_suffix(ext).name
        outfile = outdir / fname
        if outfile.exists() and not args.overwrite:
            continue
        bpemb.save_word2vec_format(str(outfile), binary=binary)
        emb = load_word2vec_file(outfile)
        id_emb = load_word2vec_file(spm_id_emb)
        id_emb_count = 0
        for piece, vocab_entry in emb.vocab.items():
            assert (emb.vectors[new_vocab[piece].index] == emb[piece]).all()
            assert (emb.vectors[emb.vocab[piece].index] == emb[piece]).all()
            str_id = str(piece2id[piece])
            if str_id in id_emb:
                assert (emb[piece] == id_emb[str_id]).all()
                id_emb_count += 1
        if id_emb_count == len(id_emb.vocab):
            print(outfile, "checks passed")
        else:
            print('vocab size mismatch', id_emb_count, len(id_emb.vocab))


def main():
    args = argsparser().parse_args()
    convert_emb(args.spm_id_emb, args)


if __name__ == "__main__":
    main()
