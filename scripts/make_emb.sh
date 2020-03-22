#! /usr/bin/env bash
#$ -cwd
#$ -j y
#$ -o ~/uge/\$JOB_ID
# set -uo pipefail

. ~/conda/bin/activate pt

date=2020-03-20
basedir=~/git/CORD-19/scripts
datadir=$basedir/data/$date
encdir=$basedir/views/$date/enc
glove_dir=$basedir/views/$date/glove_out
bio=biorxiv_medrxiv
comm=comm_use_subset
noncomm=noncomm_use_subset
custom=custom_license

mkdir -p $encdir
mkdir -p $glove_dir
cd $basedir

_case=$1
model_type=$2
vocab_size=$3
emb_dim=$4
emb_type=$5

if [[ $emb_type == 'phrase_emb' ]]; then
	split_by_whitespace='false'
else
	emb_type='subword_emb'
	split_by_whitespace='true'
fi

# extract text from json files using jq
text_dir=$basedir/views/$date
text_prefix=CORD-19.${date}.text
if [ ! -f $text_dir/${text_prefix}.cased ]; then
	for category in $bio $comm $noncomm $custom; do
		cat $datadir/$category/$category/*.json 
	done | jq '.body_text[].text' > $text_dir/${text_prefix}.body_text.cased
	for category in $bio $comm $noncomm; do
		cat $datadir/$category/$category/*.json 
	done | jq '.abstract[].text' > $text_dir/${text_prefix}.abstract.cased
	cat $text_dir/${text_prefix}.*.cased > $text_dir/${text_prefix}.cased
fi

# make uncased version of extracted text
if [ ! -f $text_dir/${text_prefix}.uncased ]; then
	sed "s/[[:upper:]]*/\L&/g" $text_dir/${text_prefix}.cased > $text_dir/${text_prefix}.uncased
fi

# train sentencepiece model
emb_type_prefix=CORD-19.${date}.${emb_type}
input=views/$date/${text_prefix}.${_case}
model_prefix=spm/${emb_type_prefix}.${_case}.${model_type}.vs${vocab_size}
model=${model_prefix}.model
if [ ! -f $model ]; then
	python sentencepiece_train.py \
		--input=$input \
		--model_prefix=$model_prefix \
		--model_type=$model_type \
		--vocab_size=${vocab_size} \
		--split_by_whitespace=$split_by_whitespace \
		--hard_vocab_limit=false \
		--max_sentence_length=1000000 \
		--max_sentencepiece_length=255 \
		--character_coverage=1.0
fi

# segment text with trained sentencepiece model
enc_file=$encdir/${emb_type_prefix}.${_case}.${model_type}.vs${vocab_size}
if [ -f $model ]; then
	if [ ! -f $enc_file ]; then
		python sentencepiece_encode.py \
			--input $input \
			--model $model \
			--output $enc_file
	fi
fi

# train glove embeddings on segmented text
glove_prefix=$glove_dir/$(basename $enc_file).d${emb_dim}
if [ ! -f ${glove_prefix}.txt ]; then
	./run_glove.sh $enc_file $glove_prefix $emb_dim 0.75
else
	echo ${glove_prefix}.txt
fi
if [ ! -f ${glove_prefix}.txt ]; then
	./run_glove.sh $enc_file $glove_prefix $emb_dim 0.70
fi
if [ ! -f ${glove_prefix}.txt ]; then
	./run_glove.sh $enc_file $glove_prefix $emb_dim 0.60
fi
if [ ! ${glove_prefix}.txt ]; then
	echo training failed: ${glove_prefix}
else
	# convert from glove file format to word2vec file format
	glove_emb_file=${glove_prefix}.w2v.txt
	if [ ! -f $glove_emb_file ]; then
		python -m gensim.scripts.glove2word2vec \
			--input ${glove_prefix}.txt \
			--output $glove_emb_file
	fi
	# order embeddings according to their sentencepiece model IDs
	# and convert to gensim binary file format (which is smaller and loads faster than .txt)
	emb_file=emb/${glove_prefix}.w2v.bin
	if [ ! -f $emb_file ]; then
		python convert_emb.py \
			--spm-id-emb $glove_emb_file \
			--vocab-file ${model_prefix}.vocab \
			--vocab-size $vocab_size
	fi
fi
