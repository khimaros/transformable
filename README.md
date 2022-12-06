# TRANSFORMABLE

simple command line utility for text with HuggingFace transformer pipelines (supports CPU inference)

## FEATURES

 - works with or without a GPU for [almost any](MODELS.md) transformer model
 - by default runs completely offline, on your local machine
 - automate an unlimited sequence of generation tasks for set-and-forget use
 - simple TOML configuration file for storing parameters and prompts
 - can be configured to auto-download models from HuggingFace
 - simple and fun command line interface

## USAGE

clone this repository.

install python requirements:

```shell
pip install -r requirements.txt
```

install git-lfs for large file storage support.

clone huggingface repositories into `~/src/huggingface.co/`:

```shell
git clone --recurse-submodules \
    https://huggingface.co/distilbert-base-uncased-distilled-squad/ \
    ~/src/huggingface.co/distilbert-base-uncased-distilled-squad/
```

generate a few responses using the default model (`distilbert-base-uncased-distilled-squad`):

```shell
python ./transformable.py -c ''
```

enable automatic fetching of a model from huggingface and use a manual seed:

```shell
python ./transformable.py \
    --download_models \
    --model=xlnet-large-cased \
    --seed=31911 \
    'The meaning of life is'
```

automatic model downloads will be stored in `~/.cache/huggingface/`

to dump the configuration for inspection, append the `--dump` flag to any command.

to repeat tasks (useful with random seed), use the `--repeat` flag.

for more detailed usage, see help:

```shell
python ./transformable.py --help
```

## CONFIGURATION

transformable tasks can be configured in TOML format.

by default, tasks will be read from `./transformable.toml` if it exists.

the config file uses the same keys as the flag names above.

see [transformable.example.toml](transformable.example.toml) for an example.

to execute a task from the config:

```shell
$ python ./transformable.py -c ./transformable.example.toml -t introductions
```

by default, the TOML section is used as the output name.

you can override config options with flags:

```shell
python ./transformable.py \
    -c ./transformable.example.toml \
    -t home-planet \
    -x 'Most humans are currently living on Mars.'
```

it is also possible to run multiple tasks in sequence:

```shell
python ./transformable.py \
    -c ./transformable.example.toml \
    -t introductions \
    -t meaning-of-life
```

to run all tasks from the config in sequence:

```shell
python ./transformable.py -c ./transformable.example.toml -a
```
