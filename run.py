from common_run import *
from get_allocator_stats import Get

import subprocess
import config

_interval_threshold = 2
_max_retry_num = 1
WORKHOME=os.environ['HOME']+'/px'

def single_run():
  # model = 'bert-large'
  model = 'bert-base'
  # model = 'Tacotron'
  # model = 'deepspeech'  #  need to set the rnn_type to 'rnn', 'lstm' or 'gru'
  # model = 'inception3'
  # model = 'nmt'
  batch_size = 32
  is_train = True
  dataset = None
  num_gpus = 1
  # rnn_type = 'rnn'
  rnn_type = None
  # gpu_mem_frac = 0.5
  gpu_mem_frac = -1

  # cfg = RunConfig(model=model, batch_size=batch_size, num_gpus=num_gpus, rnn_type=rnn_type)
  # cfg.initcmd(1, 0.3)
  # run_shell([cfg,])
  
  # run_seq(model=model, is_train=is_train, dataset=dataset, rnn_type=rnn_type, batch_size=batch_size,
  #         num_gpus=num_gpus, interference_test=True, only_interference=True, gpu_mem_frac=gpu_mem_frac)

  run_seq(model=model, is_train=is_train, dataset=dataset, rnn_type=rnn_type, batch_size=batch_size,
          num_gpus=num_gpus, interference_test=False, only_interference=False, gpu_mem_frac=gpu_mem_frac)
  # FindMaxBatchSize(model, num_gpus=num_gpus, base_bs=64, rnn_type=rnn_type, gpu_mem_frac=gpu_mem_frac)
  # BinarySearchBS(lower_bs=1, upper_bs=7, model=model, num_gpus=num_gpus, dataset=dataset,
  #                rnn_type=rnn_type, interference_test=True, only_interference=True, gpu_mem_frac=gpu_mem_frac)

  # time.sleep(60)
  # num_gpus = 2
  # BinarySearchBS(lower_bs=64, upper_bs=156, model=model, num_gpus=num_gpus, dataset=dataset,
  #                rnn_type=rnn_type, interference_test=True, only_interference=True, gpu_mem_frac=gpu_mem_frac)
  # time.sleep(60)

  # model = 'Tacotron'
  # rnn_type = None
  # BinarySearchBS(lower_bs=8, upper_bs=60, model=model, num_gpus=num_gpus, dataset=dataset,
  #                rnn_type=rnn_type, interference_test=True, only_interference=True)

  # cfg = RunConfig(model=model, is_train=is_train, num_gpus=1)
  # print(GetPerf(cfg, ['./perf_log/tmpout_1', './perf_log/tmperr_1']))

def GetDefaultBS(model, rnn_type=None):
  if rnn_type is None:
    return default_bs[model]
  else:
    return default_bs[model+'-'+rnn_type]


def FindMaxBatchSize(model_name, num_gpus=1, base_bs=None, rnn_type=None, gpu_mem_frac=-1):
  if base_bs is None:
    lower_bs = GetDefaultBS(model_name, rnn_type)
  else:
    lower_bs = base_bs

  # 1. 2*lower_bs to find upper bs
  upper_bs = 2 * lower_bs
  retry_num = 0
  while True:
    print('Run {}, {}'.format(model_name, upper_bs))
    res = run_seq(model=model_name, num_gpus=num_gpus, batch_size=upper_bs, 
                  interference_test=False, rnn_type=rnn_type, gpu_mem_frac=gpu_mem_frac)

    if res[0] == 0:
      # fail to run at this batch size
      break
    elif res[0] == 2:
      # success
      lower_bs = upper_bs
      upper_bs = 2 * lower_bs
    elif res[0] == 1:
      # partial failed
      if retry_num >= _max_retry_num:
        break
      else:
        retry_num += 1
    else:
      print('Error res code: {}'.format(res[0]))
      exit(1)

    time.sleep(30)

  # 2. binary-search the maximum batch size
  BinarySearchBS(lower_bs=lower_bs, upper_bs=upper_bs, model=model_name, num_gpus=num_gpus, rnn_type=rnn_type, gpu_mem_frac=gpu_mem_frac)

def BinarySearchBS(lower_bs, upper_bs, model, num_gpus=1, dataset=None, rnn_type=None,
                   interference_test=False, only_interference=False, gpu_mem_frac=-1):
  curr_bs = (lower_bs+upper_bs)//2
  retry_num = 0
  while True:
    print('Run {}, {}'.format(model, curr_bs))
    res = run_seq(model=model, num_gpus=num_gpus, batch_size=curr_bs, dataset=dataset, rnn_type=rnn_type,
                  interference_test=interference_test, only_interference=only_interference, gpu_mem_frac=gpu_mem_frac)

    if res[0] == 0:
      # failed
      next_bs = (lower_bs+curr_bs)//2
      if curr_bs - next_bs < _interval_threshold:
        break
      else:
        upper_bs = curr_bs
        curr_bs = next_bs
    elif res[0] == 1:
      # partial failed
      if retry_num >= _max_retry_num:
        retry_num = 0
        next_bs = (lower_bs+curr_bs)//2
        if curr_bs - next_bs < _interval_threshold:
          break
        else:
          upper_bs = curr_bs
          curr_bs = next_bs
      else:
        retry_num += 1
    else:
      next_bs = (curr_bs+upper_bs)//2
      if next_bs - curr_bs < _interval_threshold:
        break
      else:
        lower_bs = curr_bs
        curr_bs = next_bs

    time.sleep(30)
  

def multi_run():
  # Run CNN models
  num_gpus = 1
  is_train = True
  dataset = None
  rnn_type = None
  # gpu_mem_frac = 0.5
  gpu_mem_frac = -1


  # all_models = {
  #   'bert-base' : (20, 22),
  #   'vgg16' : (43, 23),
  #   'Tacotron' : (27, 54),
  #   'deepspeech' : (33, 48),
  #   'nmt' : (59, 124),
  # }
  # all_models = {
  #   'bert-base' : (18, 32),
  #   'inception3' : (32, 96),
  # }
  # all_models = {
  #   'Tacotron' : (27, 54),
  # }
  # all_models = {'resnet50', 'vgg16', 'inception3', 'alexnet'}
  # all_models = {'resnet152', 'inception4'}
  all_models = {'bert-base'}
  # for model in bert_models:
  #   if not batch_sizes.__contains__(model):
  #     continue
  #   BinarySearchBS(lower_bs=batch_sizes[model][0], upper_bs=batch_sizes[model][1], 
  #                  model=model, num_gpus=num_gpus, dataset=dataset,
  #                  rnn_type=rnn_type, interference_test=True, only_interference=True)

  #   time.sleep(120)
  #   num_gpus = 2
  #   BinarySearchBS(lower_bs=batch_sizes[model][0], upper_bs=batch_sizes[model][1], 
  #                  model=model, num_gpus=num_gpus, dataset=dataset,
  #                  rnn_type=rnn_type, interference_test=True, only_interference=True)
    # FindMaxBatchSize(model_name=model, num_gpus=num_gpus)

  # for model, bss in all_models.items():
  #   FindMaxBatchSize(model, num_gpus=num_gpus, base_bs=bss[0], gpu_mem_frac=gpu_mem_frac)

  # for model, bss in all_models.items():
  for model in all_models:
    if model == 'deepspeech':
      rnn_type = 'rnn'
    else:
      rnn_type = None
    
    max_in_uses = []
    alloc_ranges = []
    for i in range(10):
      server = launch_grpc_server(gpu_id=i%2+2)
      time.sleep(5)
      run_seq(model=model, is_train=is_train, dataset=dataset, rnn_type=rnn_type,
              num_gpus=num_gpus, interference_test=True, only_interference=True, gpu_mem_frac=gpu_mem_frac)
      # BinarySearchBS(lower_bs=bss[0], upper_bs=bss[1], model=model, num_gpus=i+1, dataset=dataset,
      #                rnn_type=rnn_type, interference_test=True, only_interference=True, gpu_mem_frac=gpu_mem_frac)
      time.sleep(30)
      server.kill()
      filepath = WORKHOME+'/log/server.log'
      max_in_use, alloc_range = Get(filepath)
      max_in_uses.append(max_in_use)
      alloc_ranges.append(alloc_range)
      log_bfc_allocator_stats(max_in_use, alloc_range)
    log_bfc_allocator_stats(max(max_in_uses), max(alloc_ranges))
    log_bfc_allocator_stats(min(max_in_uses), min(alloc_ranges))

def grpc_server_mix_run():
  # all_models = ['resnet50', 'alexnet']
  all_models = ['resnet50', 'vgg16', 'inception3', 'alexnet', 'inception4', 'resnet152']

  n = len(all_models)

  for i in range(n-1):
    for j in range(i+1, n):
      max_in_uses = []
      alloc_ranges = []
      model1 = all_models[i]
      model2 = all_models[j]
      for r in range(10):        
        server = launch_grpc_server(gpu_id=r%2+2)
        time.sleep(5)
        cfg1 = RunConfig(model=model1, num_gpus=1)
        cfg2 = RunConfig(model=model2, num_gpus=1)

        run_shell([cfg1, cfg2])

        time.sleep(30)
        server.kill()
        filepath = WORKHOME+'/log/server.log'
        max_in_use, alloc_range = Get(filepath)
        max_in_uses.append(max_in_use)
        alloc_ranges.append(alloc_range)
        log_bfc_allocator_stats(max_in_use, alloc_range)
      log_bfc_allocator_stats(max(max_in_uses), max(alloc_ranges))
      log_bfc_allocator_stats(min(max_in_uses), min(alloc_ranges))


def mixedrun():
  model1 = 'vgg16'
  bs1 = 18
  
  model2 = 'Tacotron'
  bs2 = 18

  num_gpus = 1
  gpu_mem_frac = -1

  if num_gpus == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(config._SINGLE_GPU_ID)
  else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

  cfg1 = RunConfig(model=model1, batch_size=bs1, num_gpus=num_gpus, gpu_mem_frac=gpu_mem_frac)
  cfg2 = RunConfig(model=model2, batch_size=bs2, num_gpus=num_gpus, gpu_mem_frac=gpu_mem_frac)

  cfg_list = [cfg1, cfg2]
  if len(cfg_list) > 1:
    config._ALLOW_GROWTH = True

  run_shell([cfg1, cfg2])

def launch_grpc_server(gpu_id=1):
  server_bin_path = WORKHOME+'/tensorflow-1.15.2/bazel-bin/tensorflow/core/distributed_runtime/rpc/grpc_testlib_server'
  exec_cmd = [
    server_bin_path,
    '--tf_jobs=localhost|29999',
    '--tf_job=localhost',
    '--tf_task=0',
    '--num_cpus=1',
    '--num_gpus=1',
  ]

  log_file = WORKHOME+'/log/server.log'
  file_out = open(log_file, 'w')
  # server_env = dict(CUDA_VISIBLE_DEVICES='1', _TF_LOG_ALLOCATOR_STATS='true')
  os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_id)
  os.environ['_TF_LOG_ALLOCATOR_STATS'] = 'true'
  os.environ['_TF_USE_ALIGNED_SCHEDULING'] = 'true' if config._TF_USE_ALIGNED_SCHEDULING else 'false'
  server_proc = subprocess.Popen(exec_cmd, shell=False, stdout=file_out.fileno(), stderr=file_out.fileno())
  return server_proc

def log_bfc_allocator_stats(max_in_use, alloc_range):
  with open('./perf_log/perf_sum.log', 'a') as fout:
    fout.write("MaxInUse: {} [{} MB] Allocation range diff: {} [{} MB]\n".format(
      max_in_use, (max_in_use >> 20), alloc_range, (alloc_range >> 20)))

def main():
  # InitPeakMem()
  # multi_run()
  grpc_server_mix_run()
  # server = launch_grpc_server(3)
  # time.sleep(10)
  # server.kill()
  # single_run()
  # mixedrun()

if __name__ == '__main__':
  main()
