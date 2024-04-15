export CUDA_VISIBLE_DEVICES=0

data_dir="../data/easycom/Easycom_N-1020_fs-48k_nrfft-257.h5"
exp_dir="./evaluation/tmp/"


epochs=2000

# fix params
eval_metric="loss"
model="dnn"
do_freqs_in=$1

if [ $do_freqs_in -eq 1 ];
then
    hidden_num=4
    hidden_dim=512
    lr=0.001
    scale_ang=32
    scale_freq=64
    B=48
    do_lars=0
    architecture=SIREN_PHASE
else
    hidden_num=4
    hidden_dim=512
    scale_ang=32
    scale_freq=64
    lr=0.001
    B=48
    do_lars=0
    architecture=SIREN_PHASE_CASCADE
fi


grid_type="regular"
ds=$2
grid_patten="checkboard"


n_mics=6

seed=42

do_physics=1
do_svect=1
do_delay=1
do_calib=1

# user params
do_bar=1

base_exp_dir="${exp_dir}/exp_${architecture}_freqs_out_seed_${seed}_ds-${ds}"

curr_exp_dir=${base_exp_dir}/freqs_${do_freqs_in}_${model}_${architecture}_phy-${do_physics}
python eusipco_main.py --exp_dir ${curr_exp_dir} --data_h5 $data_dir --seed ${seed} --epochs ${epochs} \
    --grid_type $grid_type --ds $ds --grid_pattern $grid_patten \
    --model $model --architecture $architecture --n_mics ${n_mics} \
    --batch_size $B --do_freqs_in ${do_freqs_in} --lr $lr \
    --do_svect ${do_svect} --do_delay ${do_delay} \
    --do_bar ${do_bar} --do_calib ${do_calib} \
    --hidden_dim ${hidden_dim} --hidden_num ${hidden_num} \
    --scale_ang  ${scale_ang} --scale_freq ${scale_freq} \
    --lars ${do_lars} \
    --loss_atf_lam 0  \
    --loss_mag_lam   1   \
    --loss_phase_lam 5 \
    --loss_air_lam   5 \
    --loss_caus_lam  1 \
    --loss_caus_crit "hilbert" \
    --loss_rft_lam   1 \
    --monitor_metric ${eval_metric}