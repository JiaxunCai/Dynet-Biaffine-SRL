# Dynet-Biaffine-SRL
This repository implements the semantic role labeler described in the paper [A Full End-to-End Semantic Role Labeler, Syntax-agnostic Over Syntax-aware?](https://arxiv.org/abs/1808.03815)

The codes are developed based on the [Dynet implementation of biaffine dependency parser](https://github.com/jcyk/Dynet-Biaffine-dependency-parser).

## Prerequisite
[Dynet Library](http://dynet.readthedocs.io/en/latest/)

## Usage (by examples)
### Data Preprocess

```
python preprocess-conll09.py --train /path/to/train.dataset --test /path/to/test.dataset --dev /path/to/dev.dataset
or
python preprocess-conll08.py --train /path/to/train.dataset --test /path/to/test.dataset --dev /path/to/dev.dataset
```
### Train
We use embedding pre-trained by [GloVe](https://nlp.stanford.edu/projects/glove/) (Wikipedia 2014 + Gigaword 5, 6B tokens, 100d)

```
  cd run
  python train.py --config_file ../config.cfg [--dynet-gpu]
```

### Test
```
  cd run
  python test.py --config_file ../config.cfg [--dynet-gpu]
```

All configuration options (see in `run/config.py`) can be specified by the configuration file `config.cfg`.


