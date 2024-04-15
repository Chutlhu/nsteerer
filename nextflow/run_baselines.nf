
pyfile = file("./eusipco23/eusipco_main.py")
exp_name = "baselines"
seed=666

process model_comparison_baseline_scf {
    publishDir "./eusipco23/evaluation/$exp_name", pattern: "exp_*", mode: 'copy'
    input:
        path DATA
        each ARCHITECTURE
    output:
        path("exp_*")
    script:
        grid_type="regular"
        ds=2
        """
        uuid=\$(python -c 'import uuid, sys; print(str(uuid.uuid4())[:8])')
        tag=\$uuid
        exp_dir=exp_${ARCHITECTURE}_\${tag}
        mkdir -p \$exp_dir
        python $pyfile                      \
            --exp_dir \${exp_dir}       \
            --data_h5 ${DATA}                \
            --seed ${seed}                  \
            --grid_type $grid_type           \
            --ds $ds                         \
            --model scf                      \
            --n_mics ${N_MICS} \
        """
}

process model_comparison_baseline_siren_freqs_in {
    publishDir "./eusipco23/evaluation/$exp_name", pattern: "exp_*", mode: 'copy'
    input:
        path DATA
        each ARCHITECTURE
        val EPOCHS
        each N_MICS
        each DO_PHYSICS
        each LOSS_TYPE
    output:
        path("exp_*")
    script:
        freqs_in=1
        grid_type="regular"
        ds=2
        hidden_dim=64
        hidden_num=3
        lr=0.01
        CAUS_CRIT="slope"
        template 'main_losses.sh'
}


process model_comparison_baseline_siren_freqs_out {
    publishDir "./eusipco23/evaluation/$exp_name", pattern: "exp_*", mode: 'copy'
    input:
        path DATA
        each ARCHITECTURE
        val EPOCHS
        each N_MICS
        each DO_PHYSICS
        each LOSS_TYPE
    output:
        path("exp_*")
    script:
        freqs_in=0
        grid_type="regular"
        ds=2
        hidden_dim=256
        hidden_num=2
        lr=0.001
        CAUS_CRIT="slope"
        template 'main_losses.sh'
}

DATA = Channel.fromPath("./data/easycom/Easycom_N-1020_fs-48k_nrfft-257.h5")
ARCHITECTURE = ["SIREN"]
fARCHITECTURE = ["SIREN", "fSIREN"]
N_MICS = [1, 4]
DO_PHYSICS = [0, 1]
LOSS_TYPE = ["atf"]


workflow {
    main:
        model_comparison_baseline_scf(DATA, "SCF",  N_MICS)
        // model_comparison_baseline_siren_freqs_in(DATA, ARCHITECTURE,  2000, N_MICS, DO_PHYSICS, LOSS_TYPE, CAUS_CRIT)
        // model_comparison_baseline_siren_freqs_out( DATA, fARCHITECTURE, 2000, N_MICS, DO_PHYSICS, LOSS_TYPE, CAUS_CRIT)
}