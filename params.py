import sys
import argparse

def get_argument_parser():
  parser = argparse.ArgumentParser()

  parser.add_argument('--action', type=str, choices={'grpc_server', 'local', 'get_perf'})
  
  # train params
  parser.add_argument('--is_train', type=bool, default=True, help='Train or inference')
  parser.add_argument('--fp16', action='store_true',
                      help='whether use fp16 to train, if you want to use ZeRO (currently) you must use this mode')
  parser.add_argument('--model', type=str, help='which model to run')
  parser.add_argument('--num_gpus', type=int, help='how many GPUs to use')
  parser.add_argument('--gpu_id', type=int, default=1, help='GPU id to use')
  parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
  parser.add_argument('--num_batches', type=int, default=100, help='Number of batches')
  parser.add_argument('--gpu_mem_frac', type=float, default=-1, help='GPU memory fraction to use')
  parser.add_argument('--variable_update', type=str, default='parameter_server', help='The way to update parameters')

  # TensorFlow RNN params
  parser.add_argument('--rnn_type', type=str, default='', help='RNN structure type')

  # Pytorch language model related params
  parser.add_argument('--seq_length', type=int, default=128, help='Sequence length of language model')
  parser.add_argument('--is_pretrain', action='store_true', help='whether to use pretrained weight')
  parser.add_argument('--optimizer_name', type=str, default='Adam', help='The optimizer to use')
  
  # ZeRO params
  parser.add_argument('--stage', type=int, default=0, help='ZeRO optimization stage')
  parser.add_argument('--allgather_bucket_size', type=int, default=5e8, help='ZeRO allgather bucket size')
  parser.add_argument('--overlap_comm', type=bool, default=False,
                      help='ZeRO if overlap communication, which will increase the bucket size by 4.5x')
  parser.add_argument('--reduce_bucket_size', type=int, default=5e8, help='ZeRO reduce-scatter bucket size')
  parser.add_argument('--offload_param', type=bool, default=False, help='ZeRO offload parameters to CPU/NVME')
  parser.add_argument('--offload_optimizer', type=bool, default=False, help='ZeRO offload optimizer to CPU/NVME')

  # grpc server run config
  parser.add_argument('--num_streams', type=int, default=1, help='How many GPU streams to use')

  # perf file
  parser.add_argument('--perf_file', type=str, help='The file to get performance, used in get_perf mode')

  return parser