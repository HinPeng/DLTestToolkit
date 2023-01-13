from __future__ import print_function

import os
import sys
import time

import argparse
import logging
import logger
import json
import subprocess

import numpy as np

import config


home_dir = os.environ['HOME']
WORKHOME = os.path.join(home_dir, config._PREFIX)
BERT_BASE_HOME = os.path.join(WORKHOME, 'bert_models')

_perf_offset = 0

_eval_batch_size = 64

cnn_models = ['alexnet', 'vgg16', 'resnet50', 'inception3', 'resnet152', 'inception4']
cnn_cifar10_models = ['alexnet', 'resnet20_v2', 'resnet56_v2', 'resnet110_v2']

bert_models = ['bert-tiny', 'bert-mini', 'bert-small', 'bert-medium', 'bert_base_uncased', 'bert_large_uncased']

# TODO: to be modified
bert_dirs = {'bert-tiny' : '{}/uncased_L-2_H-128_A-2'.format(BERT_BASE_HOME),
             'bert-mini' : '{}/uncased_L-4_H-256_A-4'.format(BERT_BASE_HOME),
             'bert-small' : '{}/uncased_L-4_H-512_A-8'.format(BERT_BASE_HOME),
             'bert-medium' : '{}/uncased_L-8_H-512_A-8'.format(BERT_BASE_HOME),
             'bert_base_uncased' : '{}/bert_base_uncased'.format(BERT_BASE_HOME),
             'bert_large_uncased' : '{}/bert_large_uncased'.format(BERT_BASE_HOME)}

mobile_models = ['mobilenet_v2', 'mobilenet_v3_small']
preprocess_name = {'mobilenet_v2' : 'inception_v2', 'mobilenet_v3_small' : 'inception_v3'}

default_bs = {
  'alexnet' : 512,
  'vgg16' : 64,
  'resnet50' : 64,
  'inception3' : 64,
  'resnet152' : 64,
  'inception4' : 64,
  'resnet20_v2' : 128,
  'resnet56_v2' : 128,
  'resnet110_v2' : 128,
  'bert-tiny' : 8,
  'bert-mini' : 16,
  'bert-small' : 32,
  'bert-medium' : 64,
  'bert_base_uncased' : 32,
  'bert_large_uncased' : 16,
  'mobilenet_v2' : 96,
  'mobilenet_v3_small' : 192,
  'Tacotron' : 32,
  'Tacotron-2' : 32,
  'deepspeech-rnn' : 128,
  'deepspeech-lstm' : 8,
  'deepspeech-gru' : 24,
  'nmt' : 128,
}

# maximum batch sizes when running two jobs in 1-gpu and 2-gpus
# cuda mps enabled
tj_max_batch_sizes = {
  'alexnet' : (1403, 1519),
  'vgg16' : (35, 17),
  'resnet50' : (82, 67),
  'inception3' : (40, 40),
  'bert-base' : (18, 21),
}

default_steps = {
  'cnn' : 100,
  'bert' : 100,
  'mobilenet' : 50,
  'tacotron' : 50,
  'deepspeech' : 400,
  'nmt' : 200,
}

_benchmark_cnn_dir = '{}/benchmarks/scripts/tf_cnn_benchmarks'.format(WORKHOME)
_bert_single_gpu_dir = '{}/bert'.format(WORKHOME)
_bert_multi_gpu_dir = '{}/bert_multigpu/bert'.format(WORKHOME)
_mobilenet_dir = '{}/models/research/slim'.format(WORKHOME)
_deepspeech_dir = '{}/models/research/deep_speech'.format(WORKHOME)
_tacotron2_dir = '{}/Tacotron2'.format(WORKHOME)
_deepspeed_dir = '{}/DeepSpeedExamples'.format(WORKHOME)
_deepspeed_bert_dir = '{}/bing_bert'.format(_deepspeed_dir)

peak_mem = dict()
perf_dir = './perf_log'
log_dir = os.path.join(WORKHOME, 'log')
if config._USE_CUDA_MPS:
  perf_log = '{}/{}'.format(perf_dir, 'perf_sum_mps.log')
else:
  perf_log = '{}/{}'.format(perf_dir, 'perf_sum.log')
if not os.path.exists(perf_dir):
  os.mkdir(perf_dir)
if not os.path.exists(log_dir):
  os.mkdir(log_dir)


class BertParams(object):
  def __init__(self, model):
    if model not in bert_models:
      logging.error('Please provide a bert model!')
      return
    self.model_index = -1
    if 'base' in model:
      self.model_index = 0
    elif 'large' in model:
      self.model_index = 1
    else:
      logging.error('Can not identify which bert model it is: {}'.format(model))
      return
    
    # self._type_index = {'base' : 0, 'large' : 1}
    self.vocab_size_or_config_json_file = 119547
    self.hidden_size = [768, 1024]
    self.num_hidden_layers = [12, 24]
    self.num_attention_heads = [12, 16]
    self.intermediate_size = [3072, 4096]
    self.hidden_act = 'gelu'
    self.hidden_dropout_prob = 0.1
    self.attention_probs_dropout_prob = 0.1
    self.max_position_embeddings = 512
    self.type_vocab_size = 2
    self.initializer_range = 0.02

  def __iter__(self):
    attrs = dir(self)
    for attr in attrs:
      if attr.find('__') == 0:
        continue
      if attr == 'model_index':
        continue
      value = getattr(self, attr)
      if isinstance(value, list):
        value = value[self.model_index]
      if callable(value):
        continue
      yield (attr, value)

class OptimizerParams(object):
  def __init__(self, name='Adam', lr=1e-3, weight_decay=0.01):
    self.name = name
    self.lr = lr
    self.weight_decay = weight_decay
    self.bias_correction = False

# ZeRO sub parameters
class OffloadParam():
  def __init__(self):
    self.device = 'cpu'
    self.nvme_path = "/local_nvme"
    self.pin_memory = True
    self.buffer_count = 5
    self.buffer_size = 1e8
    self.max_in_cpu = 1e9

  def __iter__(self):
    attrs = dir(self)
    for attr in attrs:
      if attr.find('__') == 0:
        continue
      value = getattr(self, attr)
      if callable(value):
        continue
      yield (attr, value)

# ZeRO sub parameters
class OffloadOptimizer():
  def __init__(self):
    self.device = 'cpu'
    self.nvme_path = '/local_nvme'
    self.pin_memory = True
    self.buffer_count = 4
    self.fast_init = False

  def __iter__(self):
    attrs = dir(self)
    for attr in attrs:
      if attr.find('__') == 0:
        continue
      value = getattr(self, attr)
      if callable(value):
        continue
      yield (attr, value)

class ZeROParams(object):
  def __init__(self, args):
    self.stage = args.stage
    self.allgather_partitions = True
    self.allgather_bucket_size : int = args.allgather_bucket_size
    self.overlap_comm = args.overlap_comm
    self.reduce_scatter = True
    self.reduce_bucket_size : int = args.reduce_bucket_size
    self.contiguous_gradients = True
    self.__is_offload_param = args.offload_param
    self.offload_param = dict(OffloadParam())
    self.__is_offload_optimizer = args.offload_optimizer
    self.offload_optimizer = dict(OffloadOptimizer())  
    self.stage3_max_live_parameters : int = 1e9
    self.stage3_max_reuse_distance : int = 1e9
    self.stage3_prefetch_bucket_size : int = args.reduce_bucket_size  # default is 5e8, make this value equal to reduce_bucket_size
    self.stage3_param_persistence_threshold : int = 1e6
    self.sub_group_size : int = 1e12
    # self.elastic_checkpoint = # Not sure whether this option exists
    self.stage3_gather_16bit_weights_on_model_save = False
    self.ignore_unused_parameters = True
    self.round_robin_gradients = False

  def __iter__(self):
    attrs = dir(self)
    for attr in attrs:
      if attr.find('__') == 0 or attr.find('_ZeROParams') == 0:
        continue
      if attr == 'offload_param' and not self.__is_offload_param:
        continue
      if attr == 'offload_optimizer' and not self.__is_offload_optimizer:
        continue
      value = getattr(self, attr)
      if callable(value):
        continue
      yield (attr, value)

class RunConfig(object):
  def __init__(self, args):
    super(RunConfig, self).__init__()

    self.model = args.model
    self.batch_size = args.batch_size
    self.num_batches = args.num_batches
    self.num_gpus = args.num_gpus
    self.gpu_mem_frac = args.gpu_mem_frac  # TODO
    self.variable_update = args.variable_update
    self.is_train = args.is_train
    self.dataset = None
    self.rnn_type = args.rnn_type
    self.seq_length = args.seq_length
    self.optimizer_params = OptimizerParams(args.optimizer_name)
    self.zero_params = ZeROParams(args)

    # deepspeed configs
    self.ds_config_fname = None
    self.bert_config_fname = None
    self.is_fp16 = args.fp16
    self.is_pretrain = args.is_pretrain
    self.is_pt = False
    self.is_ds = False
    
    # the execution commands
    self.cd_cmd = None
    self.pre_cmd = None
    self.run_cmd = None

    if self.model.startswith('ds-'):
      self.is_ds = True
      self.is_pt = True
      self.model = self.model[3:]
    elif self.model.startswith('pt-'):
      # judge if pytorch run and remove model name prefix
      self.is_pt = True
      self.model = self.model[3:]

    if self.batch_size == -1:
      if default_bs.__contains__(self.model):
        if self.rnn_type is not None:
          self.batch_size = default_bs[self.model+'-'+self.rnn_type]
        else:
          self.batch_size = default_bs[self.model]

    if self.num_batches == -1:
      if self.iscnn():
        self.num_batches = default_steps['cnn']
      elif self.isbert():
        self.num_batches = default_steps['bert']
      elif self.ismobilenet():
        self.num_batches = default_steps['mobilenet']
      elif self.istacotron():
        self.num_batches = default_steps['tacotron']
      elif self.isdeepspeech():
        self.num_batches = default_steps['deepspeech']
      elif self.isnmt():
        self.num_batches = default_steps['nmt']
      else:
        print('Unrecognized model: {}'.format(self.model))
        exit(1)

    if self.dataset == None:
      if self.ismobilenet():
        self.dataset = 'imagenet'

  # def __repr__(self):
  #   model_name = ''
  #   if self.is_ds:
  #     model_name = 'ds-'
  #   elif self.is_pt:
  #     model_name = 'pt-'
  #   else:
  #     model_name = 'tf-'
  #   model_name += self.model
  #   res = '{}_{}_{}_GPU{}'.format('train' if self.is_train else 'eval',
  #                                 model_name, self.batch_size, self.num_gpus)

  #   if self.is_fp16:
  #     res += '_fp16'
  #   if self.is_ptbert():
  #     res += '_{}_seq{}'.format(self.optimizer_params.name ,self.seq_length)
  #     if self.zero_params.stage > 0:
  #       res += '_zero{}_bus{:.1e}'.format(self.zero_params.stage, self.zero_params.reduce_bucket_size)
  #   if self.isrnn():
  #     res += '_{}'.format(self.rnn_type)
  #   if config._ALLOW_GROWTH:
  #     res += '_ag'

  #   return res

  def job_name(self):
    model_name = ''
    if self.is_ds:
      model_name = 'ds-'
    elif self.is_pt:
      model_name = 'pt-'
    else:
      model_name = 'tf-'
    model_name += self.model
    res = '{}_{}_{}_GPU{}'.format('train' if self.is_train else 'eval',
                                  model_name, self.batch_size, self.num_gpus)
    if self.is_fp16:
      res += '_fp16'
    if self.is_ptbert():
      res += '_{}_seq{}'.format(self.optimizer_params.name, self.seq_length)
      if self.zero_params.stage > 0:
        # zero stage with bucket size
        # with an assumption that allgather_bucket_size == reduce_scatter_size
        res += '_zero{}_bus{:.1e}'.format(self.zero_params.stage, self.zero_params.reduce_bucket_size)
      if self.zero_params.overlap_comm:
        res += '_ovc'

    return res

  def cmd_debug(self):
    logging.debug('cd_cmd: {}'.format(self.cd_cmd))
    logging.debug('pre_cmd: {}'.format(self.pre_cmd))
    logging.debug('run_cmd: {}'.format(self.run_cmd))

  def iscnn(self):
    return self.model in cnn_models or self.model in cnn_cifar10_models

  def isrnn(self):
    return self.istacotron() or self.isdeepspeech()

  def isbert(self):
    return self.model in bert_models

  def is_tfbert(self):
    return self.isbert() and not self.is_pt

  def is_ptbert(self):
    return self.isbert() and self.is_pt

  def ismobilenet(self):
    return self.model in mobile_models

  def istacotron(self):
    return 'tacotron' in self.model.lower()

  def isdeepspeech(self):
    return 'deepspeech' in self.model.lower()

  def isnmt(self):
    return 'nmt' in self.model.lower()

  def init_deepspeed_config(self):
    ds_config = dict()
    ds_config['train_batch_size'] = self.batch_size
    ds_config['train_micro_batch_size_per_gpu'] = self.batch_size // self.num_gpus
    ds_config['steps_per_print'] = 1000
    ds_config['prescale_gradients'] = False
    ds_config['gradient_predivide_factor'] = 1.0  # default value
    # gradient_predivide_factor != 1.0 is not yet supported with ZeRO-2 with reduce scatter enabled

    # optimizer related configurations
    ds_config['optimizer'] = dict()
    opt_config = ds_config['optimizer']
    opt_config['type'] = self.optimizer_params.name
    opt_config['params'] = dict()
    opt_config['params']['lr'] = self.optimizer_params.lr
    opt_config['params']['weight_decay'] = self.optimizer_params.weight_decay
    opt_config['params']['bias_correction'] = self.optimizer_params.bias_correction

    ds_config['gradient_clipping'] = 1.0
    ds_config['wall_clock_breakdown'] = False
    ds_config['fp16'] = dict()
    ds_config['fp16']['enabled'] = self.is_fp16
    ds_config['fp16']['loss_scale'] = 0

    # ZeRO configurations
    ds_config['zero_optimization'] = dict(self.zero_params)

    config_dir = '{}/config'.format(_deepspeed_bert_dir)
    if not os.path.exists(config_dir):
      os.mkdir(config_dir)
    self.ds_config_fname = '{}/ds_bsz{}_config_seq{}'.format(
        config_dir, self.batch_size, self.seq_length)
    if self.is_fp16:
      self.ds_config_fname += '_fp16'
    if self.zero_params.stage > 0:
      self.ds_config_fname += '_zero{}_bus{:.1e}'.format(self.zero_params.stage, self.zero_params.reduce_bucket_size)
    self.ds_config_fname += '.json'
    with open(self.ds_config_fname, 'w') as fout:
      json.dump(ds_config, fout, indent=4)

  def init_ds_model_config(self):
    ds_model_config = dict()
    ds_model_config['name'] = 'bing_{}_seq'.format(self.model)
    ds_model_config['bert_token_file'] = '{}/{}'.format(BERT_BASE_HOME, self.model)
    ds_model_config['bert_model_file'] = '{}/{}/pt'.format(BERT_BASE_HOME, self.model)

    # Bert model configurations
    ds_model_config['bert_model_config'] = dict(BertParams(self.model))

    # dataset configurations
    ds_model_config['data'] = dict()
    data_config = ds_model_config['data']
    data_config['flags'] = dict()
    data_config['flags']['pretrain_dataset'] = True
    data_config['flags']['pretrain_type'] = 'wiki_bc'
    data_config['mixed_seq_datasets'] = dict()
    data_config['mixed_seq_datasets']['128'] = dict()
    data_config['mixed_seq_datasets']['512'] = dict()
    data_config['mixed_seq_datasets']['128']['pretrain_dataset'] = '{}/datasets/hdf5_uncase_128_mp20/wikicorpus_en'.format(home_dir)
    data_config['mixed_seq_datasets']['512']['pretrain_dataset'] = '{}/datasets/hdf5_uncase_512_mp20/wikicorpus_en'.format(home_dir)

    # mixed training configurations
    ds_model_config['mixed_seq_training'] = dict()
    train_config = ds_model_config['mixed_seq_training']
    # seq 128
    train_config['128'] = dict()
    train_config['128']['num_epochs'] = 1
    train_config['128']['warmup_proportion'] = 0.06
    train_config['128']['learning_rate'] = 11e-3
    train_config['128']['num_workers'] = 4
    train_config['128']['async_worker'] = True
    train_config['128']['decay_rate'] = 0.90
    train_config['128']['decay_step'] = 250
    train_config['128']['total_training_steps'] = 7500
    # seq 512
    train_config['512'] = dict()
    train_config['512']['num_epochs'] = 1
    train_config['512']['warmup_proportion'] = 0.02
    train_config['512']['learning_rate'] = 2e-3
    train_config['512']['num_workers'] = 4
    train_config['512']['async_worker'] = True
    train_config['512']['decay_rate'] = 0.90
    train_config['512']['decay_step'] = 150
    train_config['512']['total_training_steps'] = 7500

    config_dir = '{}/config'.format(_deepspeed_bert_dir)
    if not os.path.exists(config_dir):
      os.mkdir(config_dir)
    self.bert_config_fname = '{}/{}_nvidia_data.json'.format(config_dir, self.model)
    if not os.path.exists(self.bert_config_fname):
      with open(self.bert_config_fname, 'w') as fout:
        json.dump(ds_model_config, fout, indent=4)

  def initcmd(self, i=0):
    if self.iscnn():
      self.cd_cmd = [
        'cd',
        '{}'.format(_benchmark_cnn_dir),
      ]
      self.run_cmd = [
        'python3',
        'tf_cnn_benchmarks.py',
        '--model={}'.format(self.model),
        '--batch_size={}'.format(self.batch_size),
        '--num_batches={}'.format(self.num_batches),
        '--num_gpus={}'.format(self.num_gpus),
        '--allow_growth={}'.format(config._ALLOW_GROWTH),
        '--lognode_time={}'.format(config._LOGNODE_TIME),
        '--allow_shared={}'.format(config._ALLOW_SHARE),
        '--target={}'.format(config._TARGET),
        '--build_cost_model={}'.format(config._BUILD_COST_MODEL),
        '--build_cost_model_after={}'.format(config._BUILD_COST_MODEL_AFTER),
      ]
      if self.gpu_mem_frac != -1:
        self.run_cmd += [
          '--gpu_memory_frac_for_testing={}'.format(self.gpu_mem_frac),
        ]
      if self.dataset != None:
        if self.dataset == 'cifar10':
          data_dir = 'cifar10-batches-py'
        self.run_cmd += [
          '--data_name={}'.format(self.dataset),
          '--data_dir=/data/{}'.format(data_dir),
        ]
      if not self.is_train:
        self.run_cmd += [
          '--eval=True',
          '--eval_batch_size={}'.format(_eval_batch_size),
          '--train_dir=ckpt_{}'.format(self.model),
        ]
      if self.num_gpus > 1:
        self.run_cmd += [
          '--local_parameter_device=cpu',
          '--variable_update={}'.format(self.variable_update),
        ]
    elif self.is_tfbert():
      bert_dir = ""
      if self.num_gpus == 1:
        bert_dir = _bert_single_gpu_dir
      else:
        bert_dir = _bert_multi_gpu_dir

      self.cd_cmd = [
        'cd',
        bert_dir,
      ]
      self.pre_cmd = [
        'rm',
        '-rf',
        './output/pretraining_output_{}'.format(i),
      ]
      common_cmd = [
        '--input_file={}/output/tf_examples.tfrecord'.format(_bert_single_gpu_dir),
        '--output_dir={}/output/pretraining_output_{}'.format(_bert_single_gpu_dir,i),
        '--do_train=True',
        '--do_eval=False',
        '--bert_config_file={}/tf/bert_config.json'.format(bert_dirs[self.model]),
        '--train_batch_size={}'.format(self.batch_size),
        '--max_seq_length=128',
        '--max_predictions_per_seq=20',
        '--num_train_steps={}'.format(self.num_batches),
        '--num_warmup_steps=10',
        '--learning_rate=2e-5',
        '--save_checkpoints_steps=0',
        '--allow_growth={}'.format(config._ALLOW_GROWTH),
        '--lognode_time={}'.format(config._LOGNODE_TIME),
        '--allow_shared={}'.format(config._ALLOW_SHARE),
        '--master={}'.format(config._TARGET),
        '--build_cost_model={}'.format(config._BUILD_COST_MODEL),
        '--build_cost_model_after={}'.format(config._BUILD_COST_MODEL_AFTER),
      ]
      if self.gpu_mem_frac != -1:
        common_cmd += [
          '--gpu_memory_frac_for_testing={}'.format(self.gpu_mem_frac),
        ]
      if self.num_gpus == 1:        
        self.run_cmd = [
          'python',
          'run_pretraining.py',          
        ] + common_cmd
      else:
        self.run_cmd = [
          'mpirun',
          '-np',
          '{}'.format(self.num_gpus),
          '-H',
          'localhost:4',
          '-bind-to',
          'none',
          '-map-by',
          'slot',
          '-x',
          'NCCL_DEBUG=INFO',
          '-x',
          'LD_LIBRARY_PATH',
          '-x',
          'PATH',
          '-mca',
          'pml',
          'ob1',
          '-mca',
          'btl',
          '^openib',
          'python',
          'run_pretraining_hvd.py',
        ] + common_cmd
    elif self.ismobilenet():
      self.cd_cmd = [
        'cd',
        '{}'.format(_mobilenet_dir),
      ]
      self.pre_cmd = [
        'rm',
        '-rf',
        '/tmp/train_logs',
      ]
      pre = preprocess_name[self.model]
      self.run_cmd = [
        'python3',
        'train_image_classifier.py',
        '--train_dir=/tmp/train_logs',
        '--dataset_name={}'.format(self.dataset),
        '--dataset_split_name=train',
        '--dataset_dir=/data/{}'.format(self.dataset),
        '--preprocessing_name={}'.format(pre),
        '--model_name={}'.format(self.model),
        '--num_clones={}'.format(self.num_gpus),
        '--learning_rate=0.045',
        '--label_smoothing=0.1',
        '--moving_average_decay=0.9999',
        '--learning_rate_decay_factor=0.98',
        '--max_number_of_steps={}'.format(self.num_batches),
        '--save_summaries_secs=6000',
        '--save_interval_secs=6000',
        '--log_every_n_steps=1',
        '--lognode_time={}'.format(config._LOGNODE_TIME),
        '--allow_growth={}'.format(config._ALLOW_GROWTH),
      ]
      if self.gpu_mem_frac != -1:
        self.run_cmd += [
          '--gpu_memory_frac_for_testing={}'.format(self.gpu_mem_frac),
        ]
    elif self.istacotron():
      self.cd_cmd = [
        'cd',
        '{}'.format(_tacotron2_dir),
      ]
      logs_dir = '{}/logs-{}'.format(_tacotron2_dir, self.model)
      if os.path.exists(logs_dir):
        self.pre_cmd = [
          'rm',
          '-rf',
          '{}'.format(logs_dir),
        ]
      self.run_cmd = [
        'python',
        'train.py',
        '--model={}'.format(self.model),
        '--tacotron_train_steps={}'.format(self.num_batches),
        '--lognode_time={}'.format(config._LOGNODE_TIME),
        '--allow_growth={}'.format(config._ALLOW_GROWTH),
        '--hparams',
        'tacotron_num_gpus={},tacotron_batch_size={}'.format(self.num_gpus, self.batch_size),
      ]
      if self.gpu_mem_frac != -1:
        self.run_cmd += [
          '--gpu_memory_frac_for_testing={}'.format(self.gpu_mem_frac),
        ]
    elif self.isdeepspeech():
      self.cd_cmd = [
        'cd',
        '{}'.format(_deepspeech_dir),
      ]
      self.pre_cmd = [
        'rm',
        '-rf',
        '/tmp/deep_speech_model',
      ]
      self.run_cmd = [
        'python',
        'deep_speech.py',
        '--train_data_dir',
        '/data/librispeech_data/train-clean-100/LibriSpeech/train-clean-100.csv',
        '--eval_data_dir',
        '/data/librispeech_data/dev-clean/LibriSpeech/dev-clean.csv',
        '--num_gpus={}'.format(self.num_gpus),
        '--wer_threshold=0.23',
        '--seed=1',
        '--train_epochs=1',
        '--max_train_steps={}'.format(self.num_batches),
        '--batch_size={}'.format(self.batch_size),
        '--log_step_count_steps=20',
        '--rnn_type={}'.format(self.rnn_type),
        '--lognode_time={}'.format(config._LOGNODE_TIME),
      ]
      if self.gpu_mem_frac != -1:
        self.run_cmd += [
          '--gpu_memory_frac_for_testing={}'.format(self.gpu_mem_frac),
        ]
    elif self.isnmt():
      self.cd_cmd = [
        'cd',
        '{}/{}/DeepLearningExamples/TensorFlow/Translation/GNMT'.format(home_dir, WORKHOME),
      ]
      self.pre_cmd = [
        'rm',
        '-rf',
        'results',
      ]
      self.run_cmd = [
        'python',
        'nmt.py',
        '--mode=train',
        '--data_dir=/data/wmt16_de_en',
        '--output_dir=results',
        '--batch_size={}'.format(self.batch_size),
        '--num_gpus={}'.format(self.num_gpus),
        '--debug_num_train_steps={}'.format(self.num_batches),
        '--learning_rate=5e-4',
        '--allow_growth={}'.format(config._ALLOW_GROWTH),
        '--use_cudnn_lstm=True',
        '--use_fp16=True',
      ]
      if self.gpu_mem_frac != -1:
        self.run_cmd += [
          '--gpu_memory_frac_for_testing={}'.format(self.gpu_mem_frac),
        ]
      if self.num_gpus > 1:
        self.run_cmd += [
          '--all_reduce_spec=nccl',
          '--local_parameter_device=cpu',
        ]
    elif self.is_ptbert():
      self.init_ds_model_config()
      output_dir = '{}/bert_model_nvidia_data_outputs'.format(_deepspeed_bert_dir)
      if not os.path.exists(output_dir):
        os.mkdir(output_dir)
      
      # job_name = '{}_nvidia_data_{}_seq{}'.format(self.optimizer_params.name, self.batch_size, self.seq_length)      
      self.cd_cmd = [
        'cd',
        _deepspeed_bert_dir,
      ]
      self.pre_cmd = [
        'rm',
        '-rf',
        '{}/*'.format(output_dir),
      ]
      self.run_cmd = [
        'deepspeed',
        '{}/deepspeed_train.py'.format(_deepspeed_bert_dir),
        '--cf',
        self.bert_config_fname,
        '--max_seq_length',
        '{}'.format(self.seq_length),
        '--max_predictions_per_seq',
        '20',
        '--output_dir',
        output_dir,
        '--print_steps',
        '1',
        '--job_name',
        self.job_name(),
        '--data_path_prefix',
        '{}/datasets/hdf5_uncase_{}_mp20/wikicorpus_en'.format(home_dir, self.seq_length),
        '--use_nvidia_dataset',
        '--lr_schedule',
        'EE',
        '--lr_offset',
        '0.0',
        '--max_steps',
        '{}'.format(self.num_batches),
        '--max_steps_per_epoch',
        '{}'.format(self.num_batches),
        # '--use_pretrain',
      ]
      if self.is_ds:
        self.init_deepspeed_config()
        self.run_cmd += [
          '--deepspeed',
          '--deepspeed_config',
          self.ds_config_fname,
        ]
    else:
      print('Unsupported model yet: {}'.format(self.model))
      exit(1);

def InitPeakMem():
  peak_mem_f = './graph/mem.json'
  if not os.path.exists(peak_mem_f):
    print('Can\'t get peak mem as {} does not exist'.format(peak_mem_f))
    return

  mem_info = []
  with open(peak_mem_f) as fin:
    mem_info = json.load(fin)

  for it in mem_info:
    netname = it['netname'].lower()
    if '-' in netname:
      netname = '_'.join(netname.split('-'))
    peak_mem[netname] = float(it['peak_mem'])


# ----------- Get performance from stdout & stderr ------------ #

def _GetPerfOnce(filename, keywd=None, func=None, need_avg=False, pre_offset=0, suf_offset=0, filter_frac=0.0):
  perfs = []
  is_oom = False
  with open(filename) as fin:
    for line in fin:
      if 'OOM' in line or 'Segmentation' in line or 'Aborted' in line or 'core dumped' in line:
        is_oom = True  # avoid OOM in the middle of running
        break
      if keywd in line:
        # perf.append(float(line.split(split_wd)[-1].strip()))
        perf = func(line, keywd)
        if perf is not None:
          perfs.append(func(line, keywd))

  if is_oom:
    return 'OOM'
  elif len(perfs) == 0:
    return None
  else:
    # def log_perfs(perfs):
    #   logging.debug('Perfs length: {}'.format(len(perfs)))
    #   for p in perfs:
    #     logging.debug(p)
    if need_avg:
      assert len(perfs) > 2
      if filter_frac > 0:
        logging.debug('Enable perf filtering')
        avg = np.average(perfs[pre_offset : (-1 - suf_offset)])
        filter_perfs = []
        for i in range(pre_offset, len(perfs) - suf_offset):
          if abs(perfs[i] - avg) / avg > filter_frac:
            logging.debug('ignore [{}th] pref: {}'.format(i, perfs[i]))
            continue
          filter_perfs.append(perfs[i])
        # log_perfs(filter_perfs)
        # del log_perfs
        return np.average(filter_perfs)

      return np.average(perfs[pre_offset : (-1 - suf_offset)])
    else:
      return perfs[-1]

def _GetPerf(filelists, keywd=None, func=None, need_avg=False, pre_offset=0, suf_offset=0, filter_frac=0.0):
  perfs = []
  for filename in filelists:
    perf = _GetPerfOnce(filename, keywd=keywd, func=func, need_avg=need_avg, pre_offset=pre_offset, suf_offset=suf_offset, filter_frac=filter_frac)
    perfs.append(perf)

  res = None
  for p in perfs:
    if p == 'OOM':
      return None
    elif p is not None:
      res = p

  return res


def GetPerfCNN(filelists):
  def _func(line, keywd):
    return float(line.split(':')[-1].strip())

  keywd = 'total images'
  perf = _GetPerf(filelists, keywd=keywd, func=_func, need_avg=False)

  del _func
  return perf

# For Bert, DeepSpeech2, GNMT, need average
def GetPerfTFBert(filelists):
  # perf: xx global_step/sec
  def _func(line, keywd):
    return float(line.split(':')[-1].strip())

  keywd = 'INFO:tensorflow:global_step/sec'
  perf = _GetPerf(filelists, keywd=keywd, func=_func, need_avg=True)

  del _func
  return perf

# For Pytorch & DeepSpeed Bert
def GetPerfPTBert(filelists):
  # perf: xx it/s
  def _func(line, keywd):
    keywd_index = line.find(keywd)
    i = keywd_index - 1
    while i > 0 and (line[i].isdigit() or line[i] == '.'):
      i -= 1
    if i == (keywd_index - 1):
      return None
    try:
      perf = float(line[i:keywd_index])
    except ValueError:
      logging.error("Can not get perf: {}".format(line[i:keywd_index]))
      return None
    return perf

  keywd = 'it/s'
  perf = _GetPerf(filelists, keywd=keywd, func=_func, need_avg=True, pre_offset=2, suf_offset=2, filter_frac=0.1)

  del _func
  return perf

def GetPerfMobile(filelists):
  # perf: xx sec/step
  def _func(line, keywd):
    return float(line.split('(')[-1].split()[0].strip())

  keywd = 'INFO:tensorflow:global step'
  perf = _GetPerf(filelists, keywd=keywd, func=_func, need_avg=True)

  del _func
  return perf


def GetPerfTacotron(filelists):
  # perf : xx sec/step already averaged
  def _func(line, keywd):
    return float(line.split(',')[0].split('[')[-1].split()[0])

  keywd = 'sec/step'
  perf = _GetPerf(filelists, keywd=keywd, func=_func, need_avg=False)

  del _func
  return perf


def GetPerf(cfg, filelists):
  if cfg.iscnn():
    return GetPerfCNN(filelists)
  elif cfg.is_tfbert():
    return GetPerfTFBert(filelists)
  elif cfg.is_ptbert():
    return GetPerfPTBert(filelists)
  elif cfg.ismobilenet():
    return GetPerfMobile(filelists)
  elif cfg.istacotron():
    return GetPerfTacotron(filelists)
  elif cfg.isdeepspeech():
    return GetPerfBert(filelists)
  elif cfg.isnmt():
    return GetPerfBert(filelists)
  else:
    print('Unimplemented GetPerf for: {}'.format(cfg.model))
    exit(1)
  

def IsMemOversubscription(cfgs):
  return False

def run(cfgs):
  if IsMemOversubscription(cfgs):
    print('Oops, OOM')
    exit(1)

  job_num = len(cfgs)
  # gpu_mem_frac = float(1)/job_num
  # for i, cfg in enumerate(cfgs):
  #   cfg.initcmd(i+1, gpu_mem_frac-_FRAC_TO_SUBTRACT)
  tmpfouts = []
  tmpferrs = []
  stdouts = []
  stderrs = []
  for i in range(job_num):    
    tmpfouts.append('{}/{}_{}'.format(perf_dir, 'tmpout', i+1))
    tmpferrs.append('{}/{}_{}'.format(perf_dir, 'tmperr', i+1))
    tmpstdout = open(tmpfouts[i], 'w')
    tmpstderr = open(tmpferrs[i], 'w')
    stdouts.append(tmpstdout)
    stderrs.append(tmpstderr)

  procs = []
  try:
    for i, cfg in enumerate(cfgs):
      if cfg.pre_cmd != None:
        pre_proc = subprocess.Popen(cfg.pre_cmd)
        pre_proc.wait()

      print('exec {}'.format(cfg.run_cmd))
      proc = subprocess.Popen(cfg.run_cmd, shell=False,
                              stdout=stdouts[i].fileno(),
                              stderr=stderrs[i].fileno())
      procs.append(proc)
      print(proc.pid)
    
    for proc in procs:
      proc.wait()
  finally:
    for i in range(job_num):
      stdouts[i].flush()
      stderrs[i].flush()
      stdouts[i].close()
      stderrs[i].close()

    perfs = []
    for i in range(job_num):
      perf = GetPerf(cfgs[i], [tmpfouts[i], tmpferrs[i]])
      if perf == None:
        perf = 0.0
      perfs.append(perf)

    with open(perf_log, 'a') as fout:
      for cfg in cfgs:
        fout.write('{}\t'.format(cfg.job_name()))
        # fout.write('{}\t'.format(cfg))
      fout.write('\n')
      for perf in perfs:
        fout.write('\t{}\n'.format(perf))
      if len(perfs) > 1:
        fout.write('\tAvg perf: {}\n'.format(np.average(perfs)))

    return perfs

def run_shell(cfgs, save_log=False, grpc_server=False):
  job_num = len(cfgs)
  # gpu_mem_frac = float(1)/job_num
  for i, cfg in enumerate(cfgs):
    # cfg.initcmd(i+1, gpu_mem_frac-_FRAC_TO_SUBTRACT)
    cfg.initcmd(i+1)
  tmpfouts = []
  tmpferrs = []
  stdouts = []
  stderrs = []
  for i in range(job_num):
    out_filename = '{}/tmpout_{}'.format(perf_log, i+1)
    err_filename = '{}tmperr_{}'.format(perf_log, i+1)
    if save_log:
      out_filename = '{}/{}_out{}.log'.format(log_dir, cfgs[i].job_name(), i+1)
      err_filename = '{}/{}_err{}.log'.format(log_dir, cfgs[i].job_name(), i+1)
    tmpfouts.append(out_filename)
    tmpferrs.append(err_filename)
    tmpstdout = open(tmpfouts[i], 'w')
    tmpstderr = open(tmpferrs[i], 'w')
    stdouts.append(tmpstdout)
    stderrs.append(tmpstderr)

  procs = []
  if grpc_server:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
  try:
    for i, cfg in enumerate(cfgs):
      cd_cmd = ' '.join(cfg.cd_cmd)
      run_cmd = ' '.join(cfg.run_cmd)
      if cfg.pre_cmd != None:
        pre_cmd = ' '.join(cfg.pre_cmd)
        command = ' && '.join([cd_cmd, pre_cmd, run_cmd])
      else:
        command = ' && '.join([cd_cmd, run_cmd])
      print(command)
      proc = subprocess.Popen(command, shell=True, stdout=stdouts[i].fileno(), stderr=stderrs[i].fileno())
      procs.append(proc)
    for proc in procs:
      proc.wait()
  finally:
    for i in range(job_num):
      stdouts[i].flush()
      stderrs[i].flush()
      stdouts[i].close()
      stderrs[i].close()

    perfs = []
    for i in range(job_num):
      perf = GetPerf(cfgs[i], [tmpfouts[i], tmpferrs[i]])
      if perf == None:
        perf = 0.0
      perfs.append(perf)

    if not config._LOGNODE_TIME:
      with open(perf_log, 'a') as fout:
        for cfg in cfgs:
          fout.write('{}\t'.format(cfg.job_name()))
          # fout.write('{}\t'.format(cfg))
        fout.write('\n')
        for perf in perfs:
          fout.write('\t{}\n'.format(perf))
        if len(perfs) > 1:
          fout.write('\tAvg perf: {}\n'.format(np.average(perfs)))

    return perfs


def run_seq(model, batch_size=-1, num_gpus=None, is_train=True, 
            dataset=None, rnn_type=None, only_interference=False, 
            interference_test=True, run_fn=run_shell, gpu_mem_frac=-1):
  gpu_list = []
  if num_gpus is None:
    gpu_list = [1, 2]
  else:
    gpu_list.append(num_gpus)

  res = []
  for i in gpu_list:
    # if i == 1:
    #   os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(config._SINGLE_GPU_ID)
    # else:
    #   os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    cfg1 = RunConfig(model=model, batch_size=batch_size, is_train=is_train, dataset=dataset, num_gpus=i, rnn_type=rnn_type, gpu_mem_frac=gpu_mem_frac)
    cfg2 = RunConfig(model=model, batch_size=batch_size, is_train=is_train, dataset=dataset, num_gpus=i, rnn_type=rnn_type, gpu_mem_frac=gpu_mem_frac)

    if not only_interference:
      cfg_list = [cfg1,]
      config._ALLOW_GROWTH = False
      perfs = run_fn(cfg_list)
      res.append(check_perfs(perfs))

    if interference_test:
      if not only_interference:
        time.sleep(30)
      # config._ALLOW_GROWTH = True
      cfg_list = [cfg1, cfg2]
      perfs = run_fn(cfg_list)
      res.append(check_perfs(perfs))

  return res

def check_perfs(perfs):
  res = -1
  arr = np.array(perfs)
  if (arr==0.0).all():
    res = 0
  elif (arr==0.0).any():
    res = 1
  else:
    res = 2

  return res


if __name__ == '__main__':
  # model = sys.argv[1]
  # filename = sys.argv[2]

  parser = argparse.ArgumentParser()

  parser.add_argument('--model', '-m', required=True, type=str,
                      help='Model name')
  parser.add_argument('--batch_size', '-b', default=None, type=int,
                      help='Batch size')

  args = parser.parse_args()

  config = RunConfig(model=args.model, batch_size=args.batch_size)
  perf = GetPerf(config, [filename,])
  print("Model[{}]  Bs[{}]  Perf[examples/sec]: {}".format(model, config.batch_size, perf*config.batch_size))