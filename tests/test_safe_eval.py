import sys
sys.path.append("../utils")
import safe_eval
import numpy as np
"""
run this script with 
`python3 test_safe_eval.py`
or submit the same call to slurm. 
If doing parallel evals ask slurm for 1 more node 
than the value of n_partitions used.
"""

vmec_input = "../problem/input.nfp4_QH_warm_start"
y = np.array([ 0.129233876411961 ,-0.0319865686953681,-0.0121381086893075,
 -0.0771362650964359, 0.1925823353893422,-0.0031275795749591,
  0.0173702926437498, 0.0201247261440845,-0.0185875115419401,
  0.0123756117172881, 0.0127305521374446,-0.0031017155856168,
  0.1420841604088703, 0.0162961256314943,-0.0187325099720287,
  0.079217408105163 , 0.1562349241673583, 0.0152736378770026,
  0.01751080835338  , 0.0042537094024368,-0.0131023634159305,
 -0.0271955151660268, 0.0056313522365318,-0.0151677938781453])
x0 = np.array([ 0.1344559674021724 ,0.                , 0.,
 -0.0761187364908742 ,0.166350081735033 ,-0.012763446904552,
  0.                 ,0.                , 0.,
  0.                 ,0.                , 0.,
  0.1328736337280527 ,0.                , 0.,
  0.101600581306652  ,0.1669633931562144,-0.0182741962582313,
  0.                 ,0.                , 0.,
  0.                 ,0.                , 0.                ])

# test single evaluation
dim_F = 2
eval_script = "../utils/safe_eval.py"
n_partitions = 1
evaluator = safe_eval.SafeEval(dim_F,vmec_input,eval_script,n_partitions)
F = evaluator.eval(x0)
print(F)
F = evaluator.eval(y)
print(F)

# parallel test
test_2 = True
if test_2:
  # run this test on slurm
  # ask for n_partitions + 1 nodes
  Y = np.hstack((x0,x0,y,y,x0,y,x0))
  Y = Y.reshape((7,len(x0)))
  n_partitions = 3
  evaluator = safe_eval.SafeEval(dim_F,vmec_input,eval_script,n_partitions)
  F = evaluator.evalp(Y)
  print(F)
