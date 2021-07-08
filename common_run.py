from __future__ import print_function

import os
import sys
import time
import json
import subprocess

import numpy as np

import config

WORKHOME=os.environ['HOME']+'/px'

_perf_offset = 1

_eval_batch_size = 64

cnn_models = ['alexnet', 'vgg16', 'resnet50', 'inception3', 'resnet152', 'inception4']
cnn_cifar10_models = ['alexnet', 'resnet20_v2', 'resnet56_v2', 'resnet110_v2']

bert_models = ['bert-tiny', 'bert-mini', 'bert-small', 'bert-medium', 'bert-base', 'bert-large']
bert_base_dirs = {'bert-tiny' : '{}/bert/uncased_L-2_H-128_A-2'.format(WORKHOME),
                  'bert-mini' : '{}/bert/uncased_L-4_H-256_A-4'.format(WORKHOME),
                  'bert-small' : '{}/bert/uncased_L-4_H-512_A-8'.format(WORKHOME),
                  'bert-medium' : '{}/bert/uncased_L-8_H-512_A-8'.format(WORKHOME),
                  'bert-base' : '{}/bert/bert_base_uncase'.format(WORKHOME),
                  'bert-large' : '{}/bert/uncased_L-24_H-1024_A-16'.format(WORKHOME)}

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
  'bert-base' : 32,
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

peak_mem = dict()
perf_dir = './perf_log'
if config._USE_CUDA_MPS:
  perf_log = '{}/{}'.format(perf_dir, 'perf_sum_mps.log')
else:
  perf_log = '{}/{}'.format(perf_dir, 'perf_sum.log')
if not os.path.exists(perf_dir):
  os.mkdir(perf_dir)


class RunConfig(object):
  __slots__ = [
    'model',
    'batch_size',
    'num_batches',
    'num_gpus',
    'gpu_mem_frac',
    'variable_update',
    'is_train',
    'dataset',
    'rnn_type', # for deepspeech
    'cd_cmd',
    'pre_cmd',
    'run_cmd',
  ]

  def __init__(self, **kwargs):
    super(RunConfig, self).__init__()

    self.model = None
    self.batch_size = -1
    self.num_batches = -1
    self.num_gpus = 1
    self.gpu_mem_frac = -1
    # self.variable_update = 'parameter_server'
    self.variable_update = 'replicated'
    self.is_train = True
    self.dataset = None   # 'imagenet' or 'cifar10'
    self.rnn_type = None

    self.cd_cmd = None
    self.pre_cmd = None
    self.run_cmd = None

    for f, v in kwargs.items():
      setattr(self, f, v)

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

  def __repr__(self):
    if self.num_gpus == 1:
      res = '{}_{}_{}_{}_{}'.format('train' if self.is_train else 'eval',
                                  self.dataset, self.model, self.batch_size, self.gpu_mem_frac)
    else:
      res = '{}_{}_{}_{}_{}_{}GPUs_{}'.format('train' if self.is_train else 'eval',
                                            self.dataset, self.model, self.batch_size, self.gpu_mem_frac,
                                            self.num_gpus, self.variable_update)
    if self.rnn_type is not None:
      res += '_{}'.format(self.rnn_type)
    if config._ALLOW_GROWTH:
      res += '_ag'

    return res

  def iscnn(self):
    return self.model in cnn_models or self.model in cnn_cifar10_models

  def isbert(self):
    return self.model in bert_models

  def ismobilenet(self):
    return self.model in mobile_models

  def istacotron(self):
    return 'tacotron' in self.model.lower()

  def isdeepspeech(self):
    return 'deepspeech' in self.model.lower()

  def isnmt(self):
    return 'nmt' in self.model.lower()

  def initcmd(self, i):
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
    elif self.isbert():
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
        '--input_file={}/output/tf_examples.tfrecord'.format(bert_dir),
        '--output_dir={}/output/pretraining_output_{}'.format(bert_dir, i),
        '--do_train=True',
        '--do_eval=False',
        '--bert_config_file={}/bert_config.json'.format(bert_base_dirs[self.model]),
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
        '{}/DeepLearningExamples/TensorFlow/Translation/GNMT'.format(WORKHOME),
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

def _GetPerfOnce(filename, keywd=None, func=None, need_avg=False):
  perf = []
  is_oom = False
  with open(filename, encoding='utf-8') as fin:
    lines = fin.readlines()
    for line in lines:
      # if 'OOM' in line or 'Segmentation' in line or 'Aborted' in line or 'core dumped' in line:
      #   is_oom = True  # avoid OOM in the middle of running
      #   break
      if keywd in line:
        # perf.append(float(line.split(split_wd)[-1].strip()))
        # print('keywd: {}'.format(keywd))
        # print(line)
        # break
        perf.append(func(line))

  if is_oom:
    return 'OOM'
  elif len(perf) == 0:
    return None
  else:
    if need_avg:
      assert len(perf) > 2
      return np.average(perf[_perf_offset:])
    else:
      return perf[-1]

def _GetPerf(filelists, keywd=None, func=None, need_avg=False):
  perfs = []
  for filename in filelists:
    perf = _GetPerfOnce(filename, keywd=keywd, func=func, need_avg=need_avg)
    perfs.append(perf)

  res = None
  for p in perfs:
    if p == 'OOM':
      return None
    elif p is not None:
      res = p

  return res


def GetPerfCNN(filelists):
  def _func(line):
    return float(line.split(':')[-1].strip())

  keywd = 'total images'
  perf = _GetPerf(filelists, keywd=keywd, func=_func, need_avg=False)

  # for filename in filelists:
  #   perf = _GetPerfOnce(filename, keywd='total images', func=_func, need_avg=False)
  #   # perf = _GetPerfOnce(filename, _func, need_avg=False)
  #   if perf != None:
  #     del _func
  #     return perf

  del _func
  return perf

# For Bert, DeepSpeech2, GNMT, need average
def GetPerfBert(filelists):
  # perf: xx global_step/sec
  def _func(line):
    return float(line.split(':')[-1].strip())

  keywd = "INFO:tensorflow:global_step/sec"
  perf = _GetPerf(filelists, keywd=keywd, func=_func, need_avg=True)

  # for filename in filelists:
  #   perf = _GetPerfOnce(filename, keywd='INFO:tensorflow:global_step/sec', func=_func, need_avg=True)
  #   # perf = _GetPerfOnce(filename, _func, need_avg=True)
  #   if perf != None:
  #     del _func
  #     return perf

  del _func
  return perf


def GetPerfMobile(filelists):
  # perf: xx sec/step
  def _func(line):
    return float(line.split('(')[-1].split()[0].strip())

  keywd = 'INFO:tensorflow:global step'
  perf = _GetPerf(filelists, keywd=keywd, func=_func, need_avg=True)

  # for filename in filelists:
  #   perf = _GetPerfOnce(filename, keywd='INFO:tensorflow:global step', func=_func, need_avg=True)
  #   # perf = _GetPerfOnce(filename, _func, need_avg=True)
  #   if perf != None:
  #     del _func
  #     return perf

  del _func
  return perf


def GetPerfTacotron(filelists):
  # perf : xx sec/step already averaged
  def _func(line):
    return float(line.split(',')[0].split('[')[-1].split()[0])

  keywd = 'sec/step'
  perf = _GetPerf(filelists, keywd=keywd, func=_func, need_avg=False)

  del _func
  return perf


def GetPerf(cfg, filelists):
  if cfg.iscnn():
    return GetPerfCNN(filelists)
  elif cfg.isbert():
    return GetPerfBert(filelists)
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
        fout.write('{}\t'.format(cfg))
      fout.write('\n')
      for perf in perfs:
        fout.write('\t{}\n'.format(perf))
      if len(perfs) > 1:
        fout.write('\tAvg perf: {}\n'.format(np.average(perfs)))

    return perfs

def run_shell(cfgs):
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
    tmpfouts.append('{}/{}_{}'.format(perf_dir, 'tmpout', i+1))
    tmpferrs.append('{}/{}_{}'.format(perf_dir, 'tmperr', i+1))
    tmpstdout = open(tmpfouts[i], 'w')
    tmpstderr = open(tmpferrs[i], 'w')
    stdouts.append(tmpstdout)
    stderrs.append(tmpstderr)

  procs = []
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
          fout.write('{}\t'.format(cfg))
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
  model = sys.argv[1]
  filename = sys.argv[2]

  config = RunConfig(model=model)
  perf = GetPerf(config, [filename,])
  print("Model[{}]  Bs[{}]  Perf[examples/sec]: {}".format(model, config.batch_size, perf*config.batch_size))