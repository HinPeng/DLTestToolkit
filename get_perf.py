from common_run import RunConfig, GetPerf

import sys

model = 'nmt'
cfg = RunConfig(model=model)

print(GetPerf(cfg, [sys.argv[1]]))