"""
For safe evaluation of the qhprob
""" 
import sys
import pickle
sys.path.append("../../problem")
sys.path.append("../../utils")
import qh_prob1

# load the point
infile = sys.argv[1]
indata = pickle.load(open(infile,'rb'))
vmec_input = indata['vmec_input']
x = indata['x']
# do the evaluation
prob = qh_prob1.QHProb1(vmec_input,n_partitions=1)
F = prob.eval(x)
# write to output
indata['F'] = F
pickle.dump(indata,open(infile,"wb"))
