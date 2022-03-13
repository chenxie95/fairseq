# ref: https://github.com/pytorch/fairseq/tree/main/examples/hubert/simple_kmeans
# python dump_mfcc_feature.py /userhome/user/chenxie95/github/fairseq/examples/wav2vec/manifest960/ train 8 0 librispeech960h_feature_mfcc
# python dump_km_label.py librispeech960h_feature_mfcc_local train librispeech960h_feature_mfcc_kmeans 8 0 librispeech960h_feature_mfcc_kmeans_label

# MFCC feature extraction ->  K-means clusterinag -> K-means application
# for train split
mpirun --allow-run-as-root -np 8 python extract_mfcc.py --split train
python learn_kmeans.py librispeech960h_feature_mfcc_local train 8 librispeech960h_feature_mfcc_kmeans 100 --percent 0.1
mpirun --allow-run-as-root -np 8 python kmeans_label.py --split train

# for valid split
mpirun --allow-run-as-root -np 4 python extract_mfcc.py --split valid
mpirun --allow-run-as-root -np 4 python kmeans_label.py --split valid


# merge K-means results
lab_dir=librispeech960h_feature_mfcc_kmeans_label

nshard=4
split=valid
for rank in $(seq 0 $((nshard - 1))); do cat $lab_dir/${split}_${rank}_${nshard}.km; done > $lab_dir/${split}.km

nshard=8
split=train
for rank in $(seq 0 $((nshard - 1))); do cat $lab_dir/${split}_${rank}_${nshard}.km; done > $lab_dir/${split}.km

# Create a dummy dict
n_clusters=100
for x in $(seq 0 $((n_clusters - 1))); do echo "$x 1"; done >> $lab_dir/dict.km.txt
