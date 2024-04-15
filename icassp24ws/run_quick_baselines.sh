export CUDA_VISIBLE_DEVICES=0

data_dir="../../data/easycom/Easycom_N-1020_fs-48k_nrfft-257.h5"
exp_dir="./results/super_resolution_baseline"
seed="666"
epochs=11

do_freqs_in=0

grid_type="regular"
ds=1
base_exp_dir=${exp_dir}/${grid_type}_ds-${ds}

# baseline
model="scf"
curr_exp_dir=${base_exp_dir}/${model}
python main_baselines.py \
    --exp_dir ${curr_exp_dir} \
    --data_h5 $data_dir \
    --seed ${seed} \
    --grid_type $grid_type \
    --ds $ds \
    --model $model \
    --n_mics 6