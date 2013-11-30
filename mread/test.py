import matplotlib
matplotlib.use('Agg')
import pylab
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('mathtext',fontset='cm')
rc('mathtext',rm='stix')
rc('text', usetex=True)
#add amsmath to the preamble
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amssymb,amsmath}"]
import sys

print("[%s] About to create figure..." % sys.argv[1])
sys.stdout.flush()
plt.figure()
plt.clf()
plt.title(r"$t=0$")
plt.draw()
plt.savefig("fig%s.png" % sys.argv[1])
print("[%s] Saved figure." % sys.argv[1])
