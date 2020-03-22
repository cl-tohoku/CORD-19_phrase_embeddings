#!/bin/bash
set -euo pipefail

# Makes programs, downloads sample data, trains a GloVe model, and then evaluates it.
# One optional argument can specify the language used for eval script: matlab, octave or [default] python

CORPUS=$1
SAVE_FILE=$2
VECTOR_SIZE=$3
ALPHA=$4

BUILDDIR=~/sw/GloVe/build

tmpdir=$(mktemp -d)
VOCAB_FILE=${tmpdir}/vocab.txt
COOCCURRENCE_FILE=${tmpdir}/cooccurrence.bin
COOCCURRENCE_SHUF_FILE=${tmpdir}/cooccurrence.shuf.bin
VERBOSE=2
MEMORY=10
VOCAB_MIN_COUNT=0
MAX_ITER=15
WINDOW_SIZE=15
BINARY=0
NUM_THREADS=32
X_MAX=10

echo
if [ ! -f $VOCAB_FILE ]; then
	echo "$ $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
	$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE
fi
if [ ! -f $COOCCURRENCE_FILE ]; then
	echo "$ $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE"
	$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE
fi
if [ ! -f $COOCCURRENCE_SHUF_FILE ]; then
	echo "$ $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
	$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE
fi
echo "$ $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE -alpha $ALPHA"
$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE -alpha $ALPHA
