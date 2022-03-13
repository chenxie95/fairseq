from mpi4py import MPI
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--split', default='train')
args = parser.parse_args()


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

command = "python dump_mfcc_feature.py /userhome/user/chenxie95/github/fairseq/examples/wav2vec/manifest960/ {} {} {} librispeech960h_feature_mfcc_local".format(args.split, size, rank)
print ("Running command: {}".format(command))
os.system(command)
