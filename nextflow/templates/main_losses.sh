#!/bin/bash

uuid=\$(python -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
tag=\$uuid
exp_dir=exp_${ARCHITECTURE}_\${tag}

mkdir -p \$exp_dir

seed=666

case $LOSS_TYPE in
    "atf")
        loss_atf_lam=1
        loss_mag_lam=0
        loss_phase_lam=0
        loss_air_lam=0
        loss_caus_lam=0
        loss_rft_lam=0
        ;;
    "mag_phase")
        loss_atf_lam=0
        loss_mag_lam=1
        loss_phase_lam=10
        loss_air_lam=0
        loss_caus_lam=0
        loss_rft_lam=0
        ;;
    "mag_phase_air")
        loss_atf_lam=0
        loss_mag_lam=1
        loss_phase_lam=10
        loss_air_lam=1
        loss_caus_lam=0
        loss_rft_lam=0
        ;;
    "mag_phase_air_caus")
        loss_atf_lam=0
        loss_mag_lam=1
        loss_phase_lam=10
        loss_air_lam=1
        loss_caus_lam=1
        loss_rft_lam=0
        ;;
    "mag_phase_air_caus_svec")
        loss_atf_lam=0
        loss_mag_lam=1
        loss_phase_lam=10
        loss_air_lam=1
        loss_caus_lam=1
        loss_rft_lam=1
        ;;
esac

export CUDA_VISIBLE_DEVICES=0
python $pyfile                      \
    --exp_dir \${exp_dir}            \
    --data_h5 ${DATA}                \
    --seed 666                   \
    --epochs ${EPOCHS}               \
    --grid_type $grid_type           \
    --ds $ds                         \
    --model "dnn"                 \
    --architecture ${ARCHITECTURE} \
    --n_mics ${N_MICS} \
    --batch_size ${batch_size}   \
    --do_freqs_in ${freqs_in} \
    --lr ${lr} \
    --lars 0   \
    --do_svect ${DO_PHYSICS} \
    --do_delay ${DO_PHYSICS} \
    --do_bar   ${DO_BAR}     \
    --do_calib ${DO_CALIB}   \
    --hidden_dim ${hidden_dim} \
    --hidden_num ${hidden_num} \
    --scale_ang  ${scale_ang}  \
    --scale_freq ${scale_freq}  \
    --loss_atf_lam   \$loss_atf_lam   \
    --loss_mag_lam   \$loss_mag_lam   \
    --loss_phase_lam \$loss_phase_lam   \
    --loss_air_lam   \$loss_air_lam   \
    --loss_caus_lam  \$loss_caus_lam   \
    --loss_caus_crit ${CAUS_CRIT} \
    --loss_rft_lam   \$loss_rft_lam   \
    --monitor_metric loss