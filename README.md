# CORD-19 Phrase Embeddings

This repository provides phrase and subword embeddings, as well as corresponding SentencePiece segmentation models, trained on the [CORD-19](https://pages.semanticscholar.org/coronavirus-research) collection of COVID-19 research papers. Phrase embeddings allow querying for similar phrases, e.g.:

```python
>>> cord19_emb.most_similar('▁infection▁control▁measures')
[('▁infection▁control▁practices', 0.8876594305038452),
 ('▁infection▁control▁precautions', 0.8097165822982788),
 ('▁infection▁control', 0.7622669339179993),
 ('▁infection▁control▁procedures', 0.7612308263778687),
 ('▁infection▁prevention▁and▁control', 0.7595920562744141),
 ('▁prevention▁and▁control▁measures', 0.7420486211776733),
 ('▁adherence▁to', 0.7354915142059326),
 ('▁public▁health▁measures', 0.7327624559402466),
 ('▁control▁measures', 0.730279803276062),
 ('▁biosecurity▁measures', 0.728427529335022)]
```

## How to use

Install with:

```bash
pip install git+git://github.com/cl-tohoku/CORD-19_phrase_embeddings
```

Load 100-dim case-sensitive phrase embeddings, trained on the
2020-03-20 version of CORD-10  with the SentencePiece 'unigram'
unsupervised segmentation model and a vocabulary size of 200,000:

```python
>>> from cord19_emb import CORD19Emb
>>> cord19_emb = CORD19Emb('2020-03-20.phrase_emb.uncased.unigram.vs200000.d100')
```

Available models are:
- `CORD-19.2020-03-20.phrase_emb.cased.unigram.vs200000.d100`
- `CORD-19.2020-03-20.phrase_emb.uncased.unigram.vs200000.d100`
- `CORD-19.2020-03-20.subword_emb.cased.unigram.vs100000.d100`
- `CORD-19.2020-03-20.subword_emb.uncased.unigram.vs100000.d100`

The phrase models should be useful for tasks such as document classification or clustering, while the subword models provide a segmentation closer to standard tokenization.

Segment text into subwords, words, and phrases:
```python
>>> text = 'Faced with the current large-scale public health emergency, collecting, sorting, and analyzing biomedical information related to the \"coronavirus\" should be done as quickly as possible to gain a global perspective, which is a basic requirement for strengthening epidemic control capacity.'
>>> cord19_emb.encode(text)
['▁faced',
 '▁with▁the▁current',
 '▁large',
 '-',
 'scale',
 '▁public▁health▁emergency',
 ',',
 '▁collecting',
 ',',
 '▁sorting',
 ',',
 '▁and',
 '▁analyzing',
 '▁biomedical',
 '▁information',
 '▁related▁to▁the',
 '▁"',
 'coronavirus',
 '"',
 '▁should▁be▁done',
 '▁as',
 '▁quickly',
 '▁as▁possible',
 '▁to▁gain',
 '▁a▁global',
 '▁perspective',
 ',',
 '▁which▁is▁a',
 '▁basic',
 '▁requirement▁for',
 '▁strengthening',
 '▁epidemic',
 '▁control',
 '▁capacity',
 '.']
```

The segmentation is unsupervised and purely based on frequency. Only
frequent phrases are segmented as such, while less frequent phrases like
'epidemic control capacity' are split into words.

When using phrase embeddings ('phrase_emb' in model name), you can
query similar phrases:

```python
>>> cord19_emb.most_similar('▁public▁health▁emergency')
[('▁a▁public▁health▁emergency', 0.6968450546264648),
 ('▁health▁emergency', 0.6321637630462646),
 ('▁preparedness▁and▁response', 0.6275159120559692),
 ('▁public▁health▁emergencies', 0.6274486780166626),
 ('▁in▁the▁event▁of▁a', 0.6201773285865784),
 ('▁emergency▁response', 0.6147885918617249),
 ('▁infectious▁disease▁outbreak', 0.5898448824882507),
 ('▁response▁plan', 0.5878373384475708),
 ('▁disease▁outbreak', 0.5794155597686768),
 ('▁an▁emergency', 0.5759837627410889)]
```

When using subword embeddings ('subword_emb' in model name),
you can only query subwords or words:
```python
>>> cord19_emb.most_similar('▁coronavirus')
[('▁coronaviruses', 0.8036941289901733),
 ('▁CoV', 0.8004533052444458),
 ('▁Coronavirus', 0.7854971885681152),
 ('▁HCoV', 0.7079704999923706),
 ('CoV', 0.6921805143356323),
 ('▁SARS▁coronavirus', 0.687605082988739),
 ('▁CoVs', 0.6866652965545654),
 ('▁calicivirus', 0.667205274105072),
 ('▁human▁coronavirus', 0.6547725200653076),
 ('▁rotavirus', 0.6540536880493164)]
```

Encode text into SentencePiece IDs for performing an embedding lookup:
```python
>>> cord19_emb.encode_ids(text)
>>> ids
[12910, 62444, 373, 5, 1818, 41265, 3, 9034, ...]
>>> cord19_emb.vectors[ids].shape
(35, 300,)
```

Segment text and embed text in one go:
```python
>>> cord19_emb.embed(text).shape
(35, 300,)
```

Undo segmentation:
```python
>>> cord19_emb.decode(['▁infection▁control▁measures', '▁were▁effective'])
'infection control measures were effective'
```

Decode SentencePiece IDs:
```python
>>> cord19_emb.decode_ids([14565, 61440])
'infection control measures were effective'
```
