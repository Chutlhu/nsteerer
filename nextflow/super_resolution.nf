pyfile = file("./eusipco23/eusipco_model_tuning.py")
exp_name = "spear_tuning_on_hr_data"

process tune_best_model {
    publishDir "./eusipco23/evaluation/$exp_name", pattern: "exp_*"
    input:
        path DATA
        each FREQS
        each DS
    output:
        path("exp_*")
    script:
        """
        uuid=\$(python -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
        tag=\$uuid
        
        name=spear_tuning_freqs_$FREQS

        export CUDA_VISIBLE_DEVICES=0
        epochs=2000

        python $pyfile --n_trials $N_TRIALS --name \$name --tag \$tag \
        --data_h5 $DATA \
        --model dnn --architecture 'SIREN_PHASE_CASCADE' --freqs $FREQS \
        --svect --delay \
        --epochs \$epochs \
        --n_mics 6 \
        --ds 1 --grid_type 'regular' \
        --metric "rmse_time"
        """
}

// DATA = Channel.fromPath("./data/easycom/Easycom_N-1020_fs-48k_nrfft-257.h5")
DATA = Channel.fromPath("./data/easycom/*.h5")
FREQS = ["in", "out"]
DS = [30, 40, 50, 60, 70, 80, 90, 100]

workflow {
    main:
        tune_best_model(DATA, FREQS, N_TRIALS)
}