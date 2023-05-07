#!/usr/bin/env python

import argparse
import os
import os.path
import toml

parser = argparse.ArgumentParser(description='text to text transformer toolkit')
parser.add_argument(
        '-c', '--config', metavar='PATH', type=str, default='./transformable.toml',
        help='toml file to load task parameters from (default: "./transformable.toml")')
parser.add_argument(
        '-t', '--tasks', action='append', default=[],
        help='tasks from configuration file to execute')
parser.add_argument(
        '-n', '--name', type=str,
        help='base file name to use for generated images')
parser.add_argument(
        '-m', '--model', type=str, default='distilbert-base-uncased-distilled-squad',
        help='transformer model to use for inference (default: "distilbert-base-uncased-distilled-squad")')
parser.add_argument(
        '-p', '--num_outputs', type=int, default=1,
        help='number of images to generate per prompt (default: 1)')
parser.add_argument(
        '-i', '--num_inference_steps', type=int, default=50,
        help='number of denoising steps, higher increases quality (default: 50)')
parser.add_argument(
        '-s', '--seed', type=int, default=0,
        help='seed to use for generator (default: random)')
parser.add_argument(
        '-r', '--models_dir', type=str, default='~/src/huggingface.co/',
        help='root directory containing huggingface models (default: "~/src/huggingface.co")')
parser.add_argument(
        '-d', '--download_models', action='store_true', default=False,
        help='allow automatic downloading of models (default: False)')
parser.add_argument(
        '-o', '--output_dir', type=str, default='./output/',
        help='directory to write image output (default: "./output/")')
parser.add_argument(
        '-a', '--all_tasks', action='store_true',
        help='run all tasks from the configuration file')
parser.add_argument(
        '-l', '--list_tasks', action='store_true',
        help='list all tasks from the configuration file')
parser.add_argument(
        '--dump', action='store_true',
        help='dump configuration and exit')
parser.add_argument(
        '-k', '--kind', type=str, default='text-generation',
        choices=['text-generation', 'question-answering', 'conversational'],
        help='kind of transformer model (default="text-generation")')
parser.add_argument(
        '-j', '--repeat', type=int, default=1,
        help='repeat the specified task this number of times')
parser.add_argument(
        '-x', '--context', type=str, default='',
        help='context to use for answering the question')
parser.add_argument(
        'prompts', metavar='PROMPT', nargs='*',
        help='prompt to generate images from')
FLAGS = parser.parse_args()

# voodoo magic to find explicitly defined flags
FLAGS_SENTINEL = list()
FLAGS_SENTINEL_NS = argparse.Namespace(**{ key: FLAGS_SENTINEL for key in vars(FLAGS) })
parser.parse_args(namespace=FLAGS_SENTINEL_NS)
EXPLICIT_FLAGS = vars(FLAGS_SENTINEL_NS).items()

CONFIG_SKIP_FLAGS = ('config', 'tasks', 'dump', 'all_tasks', 'repeat', 'list_tasks', 'prompts')
CONFIG = {'DEFAULT': {}}
CONFIG_TASKS = []

if FLAGS.config:
    if os.path.exists(FLAGS.config):
        print('[*] loading configuration from', FLAGS.config)
        CONFIG = toml.load(FLAGS.config)
    for task in CONFIG:
        if task == 'DEFAULT': 
            continue
        CONFIG_TASKS.append(task)


def normalize_config(config, random_seed=False):
    if not config.get('seed') or random_seed:
        config['seed'] = int.from_bytes(os.urandom(2), 'big')


def task_config(task):
    config = {}
    config.update(CONFIG['DEFAULT'])
    if task not in CONFIG:
        print('[!] task not found in configuration file:', task)
        return config
    config.update(CONFIG[task])
    config['name'] = task

    # calculate which flags were set explicitly and override config options
    for key, value in EXPLICIT_FLAGS:
        if key in CONFIG_SKIP_FLAGS:
            continue
        if value is not FLAGS_SENTINEL:
            config[key] = value
        elif key not in config:
            config[key] = getattr(FLAGS, key)

    return config


def task_config_from_flags(prompt):
    config = {}
    config.update(CONFIG['DEFAULT'])
    for key, value in vars(FLAGS).items():
        if key in CONFIG_SKIP_FLAGS:
            continue
        config[key] = value
    config['prompts'] = [prompt]
    return config


def choose_image_path(root, basename):
    image_name = None
    i = 0
    while True:
        output_file = '%s.%d.png' % (basename, i)
        output_path = os.path.expanduser(os.path.join(root, output_file))
        if not os.path.exists(output_path):
            return output_path
        i += 1


def invoke_task(config):
    if not config.get('prompts'):
        print('[!] prompt must be defined in config or on command line, not running pipeline')
        return

    #if not config.get('name'):
    #    print('[!] --name must be specified in config or on command line, not running pipeline')
    #    return

    if config['kind'] == 'question-answering' and not config.get('context'):
        print('[!] must provide --context with question-answering transformers')
        return

    if not config.get('tokenizer'):
        config['tokenizer'] = config['model']

    local_files_only = False
    if not config.get('download_models'):
        model_path = os.path.expanduser(os.path.join(config['models_dir'], config['model']))
        tokenizer_path = os.path.expanduser(os.path.join(config['models_dir'], config['tokenizer']))
        local_files_only = True
    else:
        print('[*] will attempt to download models from huggingface')
        model_path = config['model']
        tokenizer_path = config['tokenizer']
    if 'download_models' in config:
        del config['download_models']

    if FLAGS.dump:
        print(config)
        return

    print('[*] using generator seed:', config['seed'])

    print('[*] preparing transformer pipeline from', model_path)

    import transformers

    transformers.set_seed(config['seed'])

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

    if config['kind'] == 'text-generation':
        model = transformers.AutoModelForCausalLM.from_pretrained(model_path, local_files_only=local_files_only)

    elif config['kind'] == 'question-answering':
        model = transformers.AutoModelForQuestionAnswering.from_pretrained(model_path, local_files_only=local_files_only)

    elif config['kind'] == 'conversational':
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=local_files_only)

    pipe = transformers.pipeline(config['kind'], model=model, tokenizer=tokenizer)

    print('[*] executing transformer pipeline with prompts:', config['prompts'])

    if config['kind'] == 'text-generation':
        #outputs = pipe(config['prompts'][0], max_length=30, num_return_sequences=config['num_outputs'])
        outputs = pipe(config['prompts'][0], max_length=None, max_new_tokens=30, num_return_sequences=config['num_outputs'])
        print('[*] generated text from transformer:', outputs[0]['generated_text'])

    elif config['kind'] == 'question-answering':
        outputs = pipe(question=config['prompts'][0], context=config['context'], max_length=30)
        print('[*] answer from transformer:', outputs['answer'])

    elif config['kind'] == 'conversational':
        outputs = pipe(transformers.Conversation(config['prompts'][0]))
        #print('[*] answer from transformer:', outputs['answer'])

    #os.makedirs(FLAGS.output_dir, exist_ok=True)

    print('[*] output from transformer:', outputs)


def run():
    tasks = FLAGS.tasks
    if FLAGS.all_tasks:
        tasks = CONFIG_TASKS

    if FLAGS.list_tasks:
        print('[*] listing available tasks:')
        print()
        for task in CONFIG_TASKS:
            print(task)
        print()
        return

    if not FLAGS.prompts and not tasks:
        print('[!] at least one prompt or one config/task must be provided')
        return

    if FLAGS.prompts and tasks:
        print('[!] must provide EITHER prompt arguments OR config/tasks')
        return

    if len(tasks) > 1 and FLAGS.name:
        print('[!] flag --name cannot be used with multiple tasks from config')
        return

    for j in range(FLAGS.repeat):
        for task in tasks:
            print('[*] loaded task from configuration file:', task)
            repeat = CONFIG[task].get('repeat', 1)
            for i in range(repeat):
                config = task_config(task)
                normalize_config(config, i > 0 or j > 0)
                invoke_task(config)

        for prompt in FLAGS.prompts:
            print('[*] loaded task from command line flags')
            config = task_config_from_flags(prompt)
            normalize_config(config, j > 0)
            invoke_task(config)


if __name__ == '__main__':
    run()
