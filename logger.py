import logging

logging.basicConfig(
  level    = logging.DEBUG,
  format   = '%(asctime)s  %(filename)s %(lineno)dL %(funcName)s : %(levelname)s  %(message)s',
  datefmt  = '%Y-%m-%d %H:%M:%S',
  filename = "run.log",
  filemode = 'w'
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s  %(filename)s %(lineno)dL %(funcName)s : %(levelname)s  %(message)s')
console.setFormatter(fmt)
logging.getLogger().addHandler(console)