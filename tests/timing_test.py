import time
import numpy as np
import sys
sys.path.append("../problem")
from qh_prob1 import QHProb1

vmec_input="../problem/input.nfp4_QH_warm_start"
prob = QHProb1(vmec_input,n_partitions=1,max_mode=3)
x0 = prob.x0
print(prob.dim_x)
n_evals = 10
np.random.seed(0)
Y = x0 + 1e-5*np.random.randn(n_evals,prob.dim_x)
t0 = time.time()
for y in Y:
  prob.eval(y)
print("\n\n\n")
total = time.time() - t0
per = total/n_evals
print('total time',total)
print('time per eval',per)
