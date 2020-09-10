#! /bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --job-name=JOBNAME1
#SBATCH --mem-per-cpu=2000
#SBATCH --time=1-20:00:00
#SBATCH --partition=parallel
#SBATCH --mail-type=FAIL
#SBATCH --tmp=2000

. ~/.bashrc

#export PYTHONPATH="$PYTHONPATH:$HOME/TISM_Michael/python/"
export CTAUX_CODE_PATH="$HOME/TOPOLOGICAL/CTAUXSM_V2/danny_interface"

echo $CTAUX_CODE_PATH

if [ -n "$SLURM_JOB_ID" ] ; then
    TMPDIR="/local/$SLURM_JOB_ID"
fi

if [ -z "$NSLOTS" ] ; then
export NSLOTS=24
fi
echo $NSLOTS

python - <<EOF
from __future__ import print_function
import sys,os
from std_imports import *
import matplotlib
matplotlib.use('Agg')

max_iters = 30
U = 2.0
mu = 1.0
beta = 20.
N = 24
M = 60
boundaries = "PP"
p = 1
q = 6
lambda_x = 2.00
gamma = 0.0
t = 1.
num_sweeps = 10**7
row_sym = "gridAF"
force_filling = None
solver = "CT-AUX"
covar_factor = 0.01

import os
cwd = os.getcwd()

import time
start = time.time()

import RDMFT_TISM

import glob
filelist = glob.glob('SE_iter*.npy')
last_iter = -1
for filename in filelist:
	num = int(filename.split('_iter')[1].split('.')[0])
	if num > last_iter:
		last_iter = num

if last_iter >= max_iters-1:
	pass
else:
	if last_iter == -1:
		last_iter = False
		valid = False

	RDMFT_TISM.TI_CTAUX(U,mu,beta,(N,M),boundaries,(p,q),max_iters,num_sweeps,row_sym,lambda_x=lambda_x,gamma=gamma,force_filling=force_filling,solver=solver)

#RDMFT_TISM.TI_ContinueCTAUX_SE(covar_factor=covar_factor,model=4)
#RDMFT_TISM.GenerateAllDataCTAUX()
#RDMFT_TISM.JudgeValidity()

if True:
    # Clean up files we don't want to keep for space saving reasons.

    def unlinkdan(x):
        try:
            os.unlink(x)
        except:
            pass

    for i in range(max_iters-6):
        if i % 30 != 0:
            unlinkdan('SE_iter{0}.npy'.format(i))
            unlinkdan('everything_iter{0}.npz'.format(i))
    unlinkdan('data.pickle.gz')
    unlinkdan('SE.pickle.gz')

    # Also remove the huge *.npz files
    if True:
        import glob
        filelist = glob.glob('G[kx]*.npz')
        for filename in filelist:
            unlinkdan(filename)

os.chdir(cwd)
with open('runtime.txt','wb') as file:
    file.write(str(time.time() - start))

EOF

dir=/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.0/U_2.0

mkdir $dir

dir=/data01/opticalgrid/pramod/TOPOLOGICAL/FINITE_SPIN/half_filling/gama_0.0/U_2.0/lambda_2.00_gama_0.0_OBC

mkdir $dir

cp info.dat  info.txt

mv *npy *npz  *dat $dir 
exit 0
