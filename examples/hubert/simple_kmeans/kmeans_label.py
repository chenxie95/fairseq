from mpi4py import MPI
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--split', default='train')
args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

command = "python dump_km_label.py librispeech960h_feature_mfcc_local {} librispeech960h_feature_mfcc_kmeans {} {} librispeech960h_feature_mfcc_kmeans_label".format(args.split, size, rank)
print ("Running command: {}".format(command))
os.system(command)
