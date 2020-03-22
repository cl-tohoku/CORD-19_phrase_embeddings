import re
from pathlib import Path
from typing import Sequence, Union, Set
import numpy as np

from pathlib import Path
from typing import IO


def sentencepiece_load(file):
    """Load a SentencePiece model"""
    from sentencepiece import SentencePieceProcessor
    spm = SentencePieceProcessor()
    spm.Load(str(file))
    return spm


# source: https://github.com/allenai/allennlp/blob/master/allennlp/common/file_utils.py#L147  # NOQA
def http_get_temp(url: str, temp_file: IO) -> None:
    import requests
    req = requests.get(url, stream=True)
    req.raise_for_status()
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    try:
        from tqdm import tqdm
        progress = tqdm(unit="B", total=total)
    except ImportError:
        progress = None
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            if progress is not None:
                progress.update(len(chunk))
            temp_file.write(chunk)
    if progress is not None:
        progress.close()
    return req.headers


# source: https://github.com/allenai/allennlp/blob/master/allennlp/common/file_utils.py#L147  # NOQA
def http_get(url: str, outfile: Path, ignore_tardir=False) -> None:
    import tempfile
    import shutil
    with tempfile.NamedTemporaryFile() as temp_file:
        headers = http_get_temp(url, temp_file)
        # we are copying the file before closing it, flush to avoid truncation
        temp_file.flush()
        # shutil.copyfileobj() starts at current position, so go to the start
        temp_file.seek(0)
        outfile.parent.mkdir(exist_ok=True, parents=True)
        if headers.get("Content-Type") == "application/x-gzip":
            import tarfile
            tf = tarfile.open(fileobj=temp_file)
            members = tf.getmembers()
            if len(members) != 1:
                raise NotImplementedError("TODO: extract multiple files")
            member = members[0]
            if ignore_tardir:
                member.name = Path(member.name).name
            tf.extract(member, str(outfile.parent))
            extracted_file = outfile.parent / member.name
            assert extracted_file == outfile, "{} != {}".format(
                extracted_file, outfile)
        else:
            with open(str(outfile), 'wb') as out:
                shutil.copyfileobj(temp_file, out)
    return outfile


def load_word2vec_file(word2vec_file, add_pad=False, pad="<pad>"):
    """Load a word2vec file in either text or bin format."""
    from gensim.models import KeyedVectors
    word2vec_file = str(word2vec_file)
    binary = word2vec_file.endswith(".bin")
    vecs = KeyedVectors.load_word2vec_format(word2vec_file, binary=binary)
    if add_pad:
        if pad not in vecs:
            add_embeddings(vecs, pad)
        else:
            raise ValueError("Attempted to add <pad>, but already present")
    return vecs


def add_embeddings(keyed_vectors, *words, init=None):
    import numpy as np
    from gensim.models.keyedvectors import Vocab
    if init is None:
        init = np.zeros
    vectors = keyed_vectors.vectors
    for word in words:
        keyed_vectors.vocab[word] = Vocab(count=0, index=vectors.shape[0])
        keyed_vectors.vectors = np.concatenate([
            vectors, init((1, vectors.shape[1]))])
        keyed_vectors.index2word.append(word)
    return vectors.shape[0]


class CORD19Emb():
    """
    Provides phrase and subword embeddings, as well as corresponding
    SentencePiece segmention models, trained on the CORD-19 collection
    of COVID-19 research papers.
    (https://pages.semanticscholar.org/coronavirus-research)

    # Examples

    Load 100-dim case-sensitive phrase embeddings, trained on the
    2020-03-20 version of CORD-10  with the SentencePiece 'unigram'
    unsupervised segmentation model and a vocabulary size of 200,000:
    >>> cord19_emb = CORD19Emb('2020-03-20.phrase_emb.cased.unigram.vs200000.d100')

    Segment text into subwords, words, and phrases:
    >>> text = 'Faced with the current large-scale public health emergency, collecting, sorting, and analyzing biomedical information related to the \"coronavirus\" should be done as quickly a
    ...: s possible to gain a global perspective, which is a basic requirement for strengthening epidemic control capacity.'
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

    The segmentation is unsupervised and purely based on frequency. Only
    frequent phrases are segmented as such, while less frequent phrases like
    'epidemic control capacity' are split into words.

    When using phrase embeddings ('phrase_emb' in model name), you can
    query similar phrases:
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

    Another example:
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

    When using subword embeddings ('subword_emb' in model name),
    you can only query subwords or words:
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

    Byte-pair encode text into IDs for performing an embedding lookup:
    >>> cord19_emb.encode_ids(text)
    >>> ids
    [12910, 62444, 373, 5, 1818, 41265, 3, 9034, ...]

    Byte-pair encode and embed text:
    >>> cord19_emb.embed(text).shape
    (35, 300,)

    Decode byte-pair-encoded text:
    >>> cord19_emb.decode(['▁infection▁control▁measures', '▁were▁effective'])
    'infection control measures were effective'

    Decode byte-pair IDs:
    >>> cord19_emb.decode_ids([14565, 61440])
    'infection control measures were effective'


    Parameters
    ----------

    model_name: ``str``, required
        Name of a segmentation and embedding model, e.g.:
        '2020-03-20.phrase_emb.uncased.unigram.vs200000.d100'
    cache_dir: ``Path'', optional (default = ``~/.cache/bpemb'')
        The folder in which downloaded model files will be cached.
    preprocess: ``bool'', optional (default = True)
        Whether to preprocess the text or not.
        Set to False if you have preprocessed the text already.
    encode_extra_options: ``str'' (default = None)
        Options that are directly passed to the SentencePiece encoder.
        See SentencePiece documentation for details.
    add_pad_emb: ``bool'', optional (default = False)
        Whether to add a special <pad> embedding to the byte pair
        embeddings, thereby increasing the vocabulary size to vs + 1.
        This embedding is initialized with zeros and appended to the end
        of the embedding matrix. Assuming "cord19_emb" is a model instance,
        the padding embedding can be looked up with "cord19_emb['<pad>']", or
        directly accessed with "cord19_emb.vectors[-1]".
    """
    url_tpl = "https://github.com/cl-tohoku/CORD-19_phrase_embeddings/blob/master/files/{file_name}?raw=true"
    emb_tpl = "CORD-19.{model_name}.w2v.bin"
    model_tpl = "CORD-19.{model_name}.model"
    archive_suffix = ".tar.gz"
    available_models = None

    def __init__(
            self,
            model_name,
            *,
            cache_dir: Path = Path.home() / Path(".cache/cord19_emb"),
            preprocess: bool = True,
            encode_extra_options: str = None,
            add_pad_emb: bool = False,
            segmenter_only: bool = False,
            ):
        self.model_name = model_name
        parts = model_name.split('.')
        assert len(parts) == 6, 'invalid model name: ' + model_name
        self.vocab_size = self.vs = int(parts[4][len('vs'):])
        self.dim = int(parts[5][len('d'):])
        self.cache_dir = Path(cache_dir)
        model_name_without_dim = '.'.join(parts[:-1])
        model_file = self.model_tpl.format(
            model_name=model_name_without_dim)
        self.model_file = self._load_file(model_file)
        self.spm = sentencepiece_load(self.model_file)
        if encode_extra_options:
            self.spm.SetEncodeExtraOptions(encode_extra_options)
        emb_file = self.emb_tpl.format(model_name=model_name)
        if not segmenter_only:
            self.emb_file = self._load_file(emb_file, archive=True)
            self.emb = load_word2vec_file(self.emb_file, add_pad=add_pad_emb)
            self.most_similar = self.emb.most_similar
            assert self.dim == self.emb.vectors.shape[1]
        self.do_preproc = preprocess and not '.cased.' in model_name
        self.BOS_str = "<s>"
        self.EOS_str = "</s>"
        self.BOS = self.spm.PieceToId(self.BOS_str)
        self.EOS = self.spm.PieceToId(self.EOS_str)

    def __getitem__(self, key):
        return self.emb.__getitem__(key)

    @property
    def vectors(self):
        return self.emb.vectors

    def _load_file(self, file, archive=False, cache_dir=None):
        if not cache_dir:
            if hasattr(self, "cache_dir"):
                cache_dir = self.cache_dir
            else:
                from tempfile import mkdtemp
                cache_dir = mkdtemp()
        cached_file = Path(cache_dir) / file
        if cached_file.exists():
            return cached_file
        suffix = self.archive_suffix if archive else ""
        file_url = self.url_tpl.format(file_name=file)
        print("downloading", file_url)
        return http_get(file_url, cached_file, ignore_tardir=True)

    def __repr__(self):
        return self.__class__.__name__ + \
            "(model_name={}, vs={}, dim={})".format(self.model_name, self.vocab_size, self.dim)

    def encode(
            self,
            texts: Union[str, Sequence[str]]
            ) -> Union[Sequence[str], Sequence[Sequence[str]]]:
        """Encode the supplied texts into byte-pair symbols.

        Parameters
        ----------
        texts: ``Union[str, Sequence[str]]'', required
            The text or texts to be encoded.

        Returns
        -------
            The byte-pair-encoded text.
        """
        return self._encode(texts, self.spm.EncodeAsPieces)

    def encode_ids(
            self,
            texts: Union[str, Sequence[str]]
            ) -> Union[Sequence[str], Sequence[Sequence[str]]]:
        """Encode the supplied texts into byte-pair IDs.
        The byte-pair IDs correspond to row-indices into the embedding
        matrix.

        Parameters
        ----------
        texts: ``Union[str, Sequence[str]]'', required
            The text or texts to be encoded.

        Returns
        -------
            The byte-pair-encoded text.
        """
        return self._encode(texts, self.spm.EncodeAsIds)

    def encode_with_eos(
            self,
            texts: Union[str, Sequence[str]]
            ) -> Union[Sequence[str], Sequence[Sequence[str]]]:
        """Encode the supplied texts into byte-pair symbols, adding
        an end-of-sentence symbol at the end of each encoded text.

        Parameters
        ----------
        texts: ``Union[str, Sequence[str]]'', required
            The text or texts to be encoded.

        Returns
        -------
            The byte-pair-encoded text.
        """
        return self.encode(
            texts,
            lambda t: self.spm.EncodeAsPieces(t) + [self.EOS_str])

    def encode_ids_with_eos(
            self,
            texts: Union[str, Sequence[str]]
            ) -> Union[Sequence[str], Sequence[Sequence[str]]]:
        """Encode the supplied texts into byte-pair IDs, adding
        an end-of-sentence symbol at the end of each encoded text.
        The byte-pair IDs correspond to row-indices into the embedding
        matrix.

        Parameters
        ----------
        texts: ``Union[str, Sequence[str]]'', required
            The text or texts to be encoded.

        Returns
        -------
            The byte-pair-encoded text.
        """
        return self._encode(
            texts,
            lambda t: self.spm.EncodeAsIds(t) + [self.EOS])

    def encode_with_bos_eos(
            self,
            texts: Union[str, Sequence[str]]
            ) -> Union[Sequence[str], Sequence[Sequence[str]]]:
        """Encode the supplied texts into byte-pair symbols, adding
        a begin-of-sentence and an end-of-sentence symbol at the
        begin and end of each encoded text.

        Parameters
        ----------
        texts: ``Union[str, Sequence[str]]'', required
            The text or texts to be encoded.

        Returns
        -------
            The byte-pair-encoded text.
        """
        return self._encode(
            texts,
            lambda t: (
                [self.BOS_str] + self.spm.EncodeAsPieces(t) + [self.EOS_str]))

    def encode_ids_with_bos_eos(
            self,
            texts: Union[str, Sequence[str]]
            ) -> Union[Sequence[str], Sequence[Sequence[str]]]:
        """Encode the supplied texts into byte-pair IDs, adding
        a begin-of-sentence and an end-of-sentence symbol at the
        begin and end of each encoded text.

        Parameters
        ----------
        texts: ``Union[str, Sequence[str]]'', required
            The text or texts to be encoded.

        Returns
        -------
            The byte-pair-encoded text.
        """
        return self._encode(
            texts,
            lambda t: [self.BOS] + self.spm.EncodeAsIds(t) + [self.EOS])

    def _encode(self, texts, fn):
        if isinstance(texts, str):
            if self.do_preproc:
                texts = self.preprocess(texts)
            return fn(texts)
        if self.do_preproc:
            texts = map(self.preprocess, texts)
        return list(map(fn, texts))

    def embed(self, text: str) -> np.ndarray:
        """Byte-pair encode text and return the corresponding byte-pair
        embeddings.

        Parameters
        ----------
        text: ``str'', required
            The text to encode and embed.

        Returns
        -------
        A matrix of shape (l, d), where l is the length of the byte-pair
        encoded text and d the embedding dimension.
        """
        ids = self.encode_ids(text)
        return self.emb.vectors[ids]

    def decode(
            self,
            pieces: Union[Sequence[str], Sequence[Sequence[str]]]
            ) -> Union[str, Sequence[str]]:
        """
        Decode the supplied byte-pair symbols.

        Parameters
        ----------
        pieces: ``Union[Sequence[str], Sequence[Sequence[str]]]'', required
            The byte-pair symbols to be decoded.

        Returns
        -------
            The decoded byte-pair symbols.
        """
        if isinstance(pieces[0], str):
            return self.spm.DecodePieces(pieces)
        return list(map(self.spm.DecodePieces, pieces))

    def decode_ids(self, ids):
        """
        Decode the supplied byte-pair IDs.

        Parameters
        ----------
        ids: ``Union[Sequence[int], Sequence[Sequence[int]]]'', required
            The byte-pair symbols to be decoded.

        Returns
        -------
            The decoded byte-pair IDs.
        """
        try:
            # try to decode list of lists
            return list(map(self.spm.DecodeIds, ids))
        except TypeError:
            try:
                # try to decode array
                return self.spm.DecodeIds(ids.tolist())
            except AttributeError:
                try:
                    # try to decode list of arrays
                    return list(map(self.spm.DecodeIds, ids.tolist()))
                except AttributeError:
                    # try to decode list
                    return self.spm.DecodeIds(ids)

    @staticmethod
    def preprocess(text: str) -> str:
        """
        Lowercase text when using 'uncased' models.

        Parameters
        ----------
        text: ``str'', required
            The text to be preprocessed.

        Returns
        -------
        The preprocessed text.
        """
        return text.lower()

    @property
    def pieces(self):
        return self.emb.index2word

    @property
    def words(self):
        return self.pieces

    def __getstate__(self):
        state = self.__dict__.copy()
        # the SentencePiece instance is not serializable since it is a
        # SWIG object, so we need to delete it before serializing
        state['spm'] = None
        return state

    def __setstate__(self, state):
        # load SentencePiece after the object has been unpickled
        model_file = (
            state["cache_dir"] / state['model_file'].name)
        if not model_file.exists():
            model_rel_path = model_file.name
            model_file = self._load_file(
                str(model_rel_path), cache_dir=state["cache_dir"])
        state['spm'] = sentencepiece_load(model_file)
        self.__dict__ = state


__all__ = [CORD19Emb]
