export CUDA_VISIBLE_DEVICES=0
data_dir="../data/easycom/Easycom_N-1020_fs-48k_nrfft-257.h5"
uuid=$(python -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
tag=$uuid

freqs=in
n_mics=6
bary=true
calib=true
grid_type=random
ds=90

name="spear_tuning_freqs_${freqs}_n_mics-${n_mics}_bary_calib"

python eusipco_model_tuning.py --n_trials 42 --name $name --tag $tag \
    --data_h5 $data_dir \
    --db_path "optuna.sqlite3" \
    --model dnn --architecture 'SIREN_PHASE_CASCADE' --freqs $freqs \
    --svect true --delay true   \
    --calib $calib --bary $bary \
    --ds $ds --grid_type $grid_type \
    --epochs 3000 \
    --n_mics ${n_mics} \
    --metric "rmse_time"