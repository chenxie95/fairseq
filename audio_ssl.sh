#!/bin/bash
set -x

# stage 1: pretrain
# stage 2: finetune
# stage 3: decode
stage=3

# model_name need to be [wav2vec|hubert]
model_name=hubert
if [ ${model_name} != "wav2vec" ] && [ ${model_name} !- "hubert" ]; then
    echo "model_name (${model_name}) needs to be wav2vec or hubert"
    exit
fi

# set working dir and output dir names
work_dir=/userhome/user/chenxie95/github/fairseq
exp_name=libri960h_base_debug


# directory where fairseq is installed
# e.g. in my docker image, it is /espnet/tools/fairseq
code_dir=/espnet/tools/fairseq
# code_dir=/fairseq

# log function
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# set compute resource
distributed_world_size=1
update_freq=[2]

output_dir=${work_dir}/outputs/${model_name}/${exp_name}

# for wav2vec
if [ ${model_name} == "wav2vec" ]; then
    if [ ${stage} -eq 1 ]; then
        log "Stage 1: pretrain"

        config_pretrain_dir=/userhome/user/chenxie95/github/fairseq/examples/wav2vec/config/pretraining
        config_pretrain_name=wav2vec2_base_librispeech_debug

        # specify the output directory for storing checkpoints, tensorboard log and train log
        output_dir=${work_dir}/outputs/${model_name}/${exp_name}
        if [ -d ${output_dir} ]; then
            echo "ERROR: output dir ${output_dir} exists!"
            echo "Please double-check it, or delete the output dir and run again"
            exit
        else
            mkdir -p ${output_dir}
        fi
        model_path=${output_dir}/checkpoints
        tb_path=${output_dir}/tensorboard
        log_file=${output_dir}/hydra_train.log
        mkdir -p ${model_path}
        mkdir -p ${tb_path}

        data_path=/userhome/user/chenxie95/github/fairseq/examples/wav2vec/manifest960
        train_subset=train
        valid_subset=valid

        max_tokens=1400000


        cd ${code_dir} && python fairseq_cli/hydra_train.py  \
        --config-dir ${config_pretrain_dir}  \
        --config-name ${config_pretrain_name}  \
        task.data=${data_path}  \
        checkpoint.save_dir=${model_path}  \
        common.tensorboard_logdir=${tb_path} \
        common.log_file=${log_file}  \
        distributed_training.distributed_world_size=${distributed_world_size}  \
        optimization.update_freq=${update_freq} \
        dataset.max_tokens=${max_tokens} \
        dataset.train_subset=${train_subset}  \
        dataset.valid_subset=${valid_subset} \
        common.log_interval=10 \

    fi

    if [ ${stage} -eq 2 ]; then
        log "Stage 2: finetune"

        # set finetune config
        config_finetune_dir=${work_dir}/examples/wav2vec/config/finetuning
        config_finetune_name=base_1h

        # set pretrained model
        pretrain_model_name=${work_dir}/outputs/wav2vec/libri960h_base/checkpoints/checkpoint_36_25000.pt

        # set finetune data
        finetune_data_path=/userhome/user/zsz01/repo/fairseq/examples/wav2vec/fine-tune-data/1h
        train_subset=train
        valid_subset=valid

        # set finetune output model
        finetune_data_mode=1h
        output_dir=${work_dir}/outputs/${model_name}/${exp_name}
        finetune_output_dir=${output_dir}/finetune_${finetune_data_mode}

        cd ${code_dir} && python3 fairseq_cli/hydra_train.py \
        --config-dir ${config_finetune_dir} \
        --config-name ${config_finetune_name} \
        distributed_training.distributed_port=8989 \
        task.data=${finetune_data_path} \
        model.w2v_path=${pretrain_model_name} \
        distributed_training.distributed_world_size=${distributed_world_size}  \
        optimization.update_freq=${update_freq} \
        dataset.train_subset=${train_subset}  \
        dataset.valid_subset=${valid_subset} \
        hydra.run.dir=${finetune_output_dir} \
        common.log_interval=10 \

    fi

    if [ ${stage} -eq 3 ]; then
        log "Stage 3: decode"

        # edit model config
        config_decode_dir=${work_dir}/examples/hubert/config/decode
        config_decode_name=infer_kenlm

        decode_output_dir=${work_dir}/outputs/${model_name}/${exp_name}/${data_mode}/decode

        decode_data_path=/mnt/lustre/sjtu/home/xc915/superb/dataset/librispeech_finetuning_data/valid

        subset=test
        ckpt_raw=/userhome/data/librispeech/librispeech_finetuning_data/valid
        model_path=/userhome/user/chenxie95/github/fairseq/outputs/wav2vec/libri960h_base/finetune_1h/checkpoints/
        model_size=checkpoint_best.pt
        result_path=/userhome/user/chenxie95/github/fairseq/outputs/wav2vec/libri960h_base/finetune_1h/decode

        # lexicon & ngram files
        lexicon_file=/userhome/user/chenxie95/github/fairseq/outputs/hubert/pretrained_models/librispeech_lexicon.lst
        arpa_file=/userhome/user/chenxie95/github/fairseq/outputs/hubert/pretrained_models/4-gram.arpa
        # need to copy the lm_librispeech_word_transformer.txt to dict.txt and change all words to uppercase manually
        # related files: https://github.com/flashlight/wav2letter/tree/main/recipes/sota/2019
        fairseqlm=/userhome/user/chenxie95/github/fairseq/outputs/hubert/pretrained_models/lm_librispeech_word_transformer.pt

        # use lm
        use_kenlm=false
        use_fairseqlm=true

        if ${use_kenlm}; then
            cd ${code_dir} && python3 examples/speech_recognition/infer.py \
            ${ckpt_raw} \
            --task audio_finetuning \
            --nbest 1 \
            --path ${model_path}/${model_size} \
            --gen-subset $subset \
            --results-path ${result_path} \
            --w2l-decoder kenlm \
            --lexicon ${lexicon_file} \
            --lm-model ${arpa_file} \
            --lm-weight 2 \
            --word-score -1 \
            --sil-weight 0 \
            --criterion ctc \
            --labels ltr \
            --max-tokens 4000000 \
            --post-process letter \

        elif ${use_fairseqlm}; then
            cd ${code_dir} && python3 examples/speech_recognition/infer.py \
            ${ckpt_raw} \
            --task audio_finetuning \
            --nbest 1 \
            --path ${model_path}/${model_size} \
            --gen-subset $subset \
            --results-path ${result_path} \
            --w2l-decoder fairseqlm \
            --lexicon ${lexicon_file} \
            --lm-model ${fairseqlm} \
            --lm-weight 2 \
            --word-score -1 \
            --sil-weight 0 \
            --criterion ctc \
            --labels ltr \
            --max-tokens 4000000 \
            --post-process letter \

        else
            cd ${code_dir} && python3 examples/speech_recognition/infer.py \
            ${ckpt_raw} \
            --task audio_finetuning \
            --nbest 1 \
            --path ${model_path}/${model_size} \
            --gen-subset $subset \
            --results-path ${result_path} \
            --w2l-decoder viterbi \
            --lm-weight 2 \
            --word-score -1 \
            --sil-weight 0 \
            --criterion ctc \
            --labels ltr \
            --max-tokens 4000000 \
            --post-process letter \

        fi

    fi
# for hubert
else
    if [ ${stage} -eq 1 ]; then
        log "Stage 1: pretrain"

        # do remember to comment the port assign in distributed setting
        # otherwise, it will cause issue without running init_process_group
        config_pretrain_dir=${work_dir}/examples/hubert/config/pretrain
        config_pretrain_name=hubert_base_librispeech

        # specify the output directory for storing checkpoints, tensorboard log and train log
        output_dir=${work_dir}/outputs/${model_name}/${exp_name}/
        if [ -d ${output_dir} ]; then
            echo "ERROR: output dir ${output_dir} exists!"
            echo "Please double-check it, or delete the output dir and run again"
            exit
        else
            mkdir -p ${output_dir}
        fi
        model_path=${output_dir}/checkpoints
        tb_path=${output_dir}/tensorboard
        log_file=${output_dir}/hydra_train.log
        mkdir -p ${model_path}
        mkdir -p ${tb_path}

        data_path=/userhome/user/chenxie95/github/fairseq/examples/wav2vec/manifest960
        label_path=/userhome/user/chenxie95/github/fairseq/examples/hubert/simple_kmeans/librispeech960h_feature_mfcc_kmeans_label
        train_subset=train
        valid_subset=valid

        # edit compute resource
        distributed_world_size=1
        update_freq=[2]
        max_tokens=1400000


        cd ${code_dir} && python fairseq_cli/hydra_train.py  \
        --config-dir ${config_pretrain_dir}  \
        --config-name ${config_pretrain_name}  \
        task.data=${data_path}  \
        task.label_dir=${label_path} \
        task.labels=["km"] \
        model.label_rate=100 \
        checkpoint.save_dir=${model_path}  \
        common.tensorboard_logdir=${tb_path} \
        common.log_file=${log_file}  \
        distributed_training.distributed_world_size=${distributed_world_size}  \
        optimization.update_freq=${update_freq} \
        dataset.max_tokens=${max_tokens} \
        dataset.train_subset=${train_subset}  \
        dataset.valid_subset=${valid_subset} \
        hydra.run.dir=${output_dir} \
        common.log_interval=10 \

    fi


    if [ ${stage} -eq 2 ]; then
        log "Stage 2: finetune"

        # set finetune config
        config_finetune_dir=${work_dir}/examples/hubert/config/finetune
        config_finetune_name=base_10h

        # set pretrained model
        output_dir=${work_dir}/outputs/${model_name}/${exp_name}/
        pretrain_model_name=/userhome/user/chenxie95/github/fairseq/outputs/hubert/pretrained_models/hubert_base_ls960.pt
        # pretrain_model_name=${output_dir}/checkpoints/checkpoint_36_25000.pt

        # set finetune data
        finetune_data_mode=10h
        finetune_data_path=/userhome/data/librispeech/librispeech_finetuning_data/${finetune_data_mode}

        # set finetune output model
        finetune_output_dir=${output_dir}/finetune_${finetune_data_mode}

        cd ${code_dir} && python3 fairseq_cli/hydra_train.py \
        --config-dir ${config_finetune_dir} \
        --config-name ${config_finetune_name} \
        task.data=${finetune_data_path} \
        task.label_dir=${finetune_data_path} \
        model.w2v_path=${pretrain_model_name} \
        hydra.run.dir=${finetune_output_dir} \
        common.log_interval=10 \

    fi

    if [ ${stage} -eq 3 ]; then
        log "Stage 3: decode"

        # edit model config
        config_decode_dir=${work_dir}/examples/hubert/config/decode

        finetune_data_mode=10h
        decode_output_dir=${output_dir}/finetune_${finetune_data_mode}/decode

        # lexicon & ngram for hubert
        lexicon_file=/userhome/user/chenxie95/github/fairseq/outputs/hubert/pretrained_models/librispeech_lexicon.lst
        arpa_file=/userhome/user/chenxie95/github/fairseq/outputs/hubert/pretrained_models/4-gram.arpa

        # use lm
        use_kenlm=false

        decode_data_path=/userhome/data/librispeech/librispeech_finetuning_data/valid
        # decode_model_path=/userhome/user/chenxie95/github/fairseq/outputs/hubert/pretrained_models/checkpoint_best.pt
        decode_model_path=/userhome/user/chenxie95/github/fairseq/outputs/hubert/libri960h_base_debug/finetune_10h/checkpoints/checkpoint_best.pt

        if ${use_kenlm}; then
            cd ${code_dir} && python3 examples/speech_recognition/new/infer.py \
            --config-dir ${config_decode_dir} \
            --config-name infer_kenlm \
            task.data=${decode_data_path} \
            task.normalize=false \
            common_eval.path=${decode_model_path} \
            dataset.gen_subset=test \
            decoding.lexicon=${lexicon_file} \
            decoding.lmpath=${arpa_file} \
            hydra.run.dir=${decode_output_dir}
        else
            cd ${code_dir} && python3 examples/speech_recognition/new/infer.py \
            --config-dir ${config_decode_dir} \
            --config-name infer_viterbi \
            task.data=${decode_data_path} \
            task.normalize=false \
            common_eval.path=${decode_model_path} \
            dataset.gen_subset=test \
            hydra.run.dir=${decode_output_dir}
        fi
    fi
fi

#********* info for dataset ***********
# finetune
# ├── dict.ltr.txt #! should edit by the user
# ├── train.ltr
# ├── train.tsv
# ├── valid.ltr -> ../valid_small/valid_small.ltr
# └── valid.tsv -> ../valid_small/train.tsv

# decode
# ├── test.ltr
# └── test.tsv


#********** info for tensorboard *************
# open http://localhost:6006/ to see the tensorboard
# tensorboard --logdir ${tb_path}

# cd scripts
# python average_checkpoints.py \
#     --inputs /mnt/exp/project/NMT \
#     --num-epoch-checkpoints 10 \
#     --output /mnt/exp/project/NMT

echo -e '\n'
echo "finshed!"
