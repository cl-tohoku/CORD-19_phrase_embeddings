from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='cord19_emb',
    version='0.0.1',
    description='Phrase and subword embedding trained on the CORD-19 collection of COVID-19 research papers',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/cl-tohoku/cord19-phrase-embeddings',
    author='Benjamin Heinzerling',
    author_email='benjamin.heinzerling@riken.jp',
    license='MIT',
    packages=['cord19_emb'],
    install_requires=[
        "gensim",
        "numpy",
        "requests",
        "sentencepiece",
        "tqdm"],
    zip_safe=True,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3"
    ])
