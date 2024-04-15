process create_easycom_datasets {
    publishDir "./data/eusipco23/", pattern: "*.h5", mode: 'copy'
    input:
        path EASYCOM_H5
        each N_RFFT
    output:
        path("*.h5")
    script:
        pyfile = file("./generate_easycom_data.py")
        """
        python $pyfile --data_dir $EASYCOM_H5 --n_rfft $N_RFFT
        """        
}


process best_simple_model_freqs_in {
    publishDir "./eusipco23/evaluation/$exp_name", pattern: "exp_*", mode: 'copy'
    input:
        path DATA
        each ARCHITECTURE
        val EPOCHS
        each N_MICS
        each DO_PHYSICS
        each LOSS_TYPE
        each CAUS_CRIT
        each DO_BAR
        each DO_CALIB
    output:
        path("exp_*")
    script:
        pyfile = file("./eusipco23/eusipco_main.py")
        exp_name = "best_arch_freqs_in"
        grid_type="regular"
        ds=2
        freqs_in=1
        batch_size=48
        hidden_num=4
        hidden_dim=512
        scale_ang=32
        scale_freq=64
        lr=0.001
        template 'main_losses.sh'
}


process best_simple_model_freqs_out {
    publishDir "./eusipco23/evaluation/$exp_name", pattern: "exp_*", mode: "copy"
    input:
        path DATA
        each ARCHITECTURE
        val EPOCHS
        each N_MICS
        each DO_PHYSICS
        each LOSS_TYPE
        each CAUS_CRIT
        each DO_BAR
        each DO_CALIB
    output:
        path("exp_*")
    script:
        pyfile = file("./eusipco23/eusipco_main.py")
        exp_name = "best_arch_freqs_out"
        grid_type="regular"
        ds=2
        freqs_in=0
        batch_size=18
        hidden_num=4
        hidden_dim=512
        scale_ang=32
        scale_freq=64
        lr=0.001
        template 'main_losses.sh'
}


process run_baselines {
    publishDir "./eusipco23/evaluation/$exp_name", pattern: "exp_*", mode: "copy"
    input:
        path DATA
        each ARCHITECTURE
        each N_MICS
    output:
        path("exp_*")
    script:
        pyfile = file("./eusipco23/eusipco_main.py")
        exp_name = "baseline_regular_grid"
        grid_type = "regular"
        ds=2
        DO_PHYSICS = 0
        freqs_in = 0
        lr = 0.001
        batch_size = 18
        hidden_dim = 512
        hidden_num = 4
        scale_ang=32
        scale_freq=64
        EPOCHS = 2000
        CAUS_CRIT = "slope"
        LOSS_TYPE = "atf"
        DO_CALIB = 0
        DO_BAR = 0
        template 'main_losses.sh'
}


process super_resolution_random_freqs_out {
    publishDir "./eusipco23/evaluation/$exp_name", pattern: "exp_*", mode: "copy"
    input:
        path DATA
        each ARCHITECTURE
        val EPOCHS
        each SR_DS
        each HIDDEN_DIM
        each HIDDEN_NUM
        each LOSS_TYPE
    output:
        path("exp_*")
    script:
        pyfile = file("./eusipco23/eusipco_main.py")
        exp_name = "super_resolution_freqs_out"
        grid_type="random"
        ds=SR_DS
        freqs_in=0
        batch_size=18
        scale_ang=32
        scale_freq=64
        lr=0.001

        N_MICS = 6
        hidden_num=HIDDEN_NUM
        hidden_dim=HIDDEN_DIM
        DO_PHYSICS = 1
        DO_CALIB = 1
        DO_BAR = 1
        CAUS_CRIT = "hilbert"

        template 'main_losses.sh'
}

process super_resolution_random_freqs_in {
    publishDir "./eusipco23/evaluation/$exp_name", pattern: "exp_*", mode: "copy"
    input:
        path DATA
        each ARCHITECTURE
        val EPOCHS
        each SR_DS
        each HIDDEN_DIM
        each HIDDEN_NUM
        each LOSS_TYPE
    output:
        path("exp_*")
    script:
        pyfile = file("./eusipco23/eusipco_main.py")
        exp_name = "super_resolution_freqs_in"
        grid_type="random"
        ds=SR_DS
        freqs_in=1
        batch_size=48
        scale_freq=64
        scale_ang=32
        lr=0.001

        hidden_num=HIDDEN_NUM
        hidden_dim=HIDDEN_DIM
        DO_CALIB = 1
        N_MICS = 6
        DO_PHYSICS = 1
        DO_BAR = 1
        CAUS_CRIT = "hilbert"

        template 'main_losses.sh'
}

process super_resolution_random_grid_baselines {
    publishDir "./eusipco23/evaluation/$exp_name", pattern: "exp_*", mode: "copy"
    input:
        path DATA
        each ARCHITECTURE
        each SR_DS
    output:
        path("exp_*")
    script:
        pyfile = file("./eusipco23/eusipco_main.py")
        exp_name = "super_resolution_baseline"
        grid_type = "random"
        ds=SR_DS
        DO_PHYSICS = 0
        freqs_in = 0
        lr = 0.001
        batch_size = 18
        hidden_dim = 512
        hidden_num = 4
        scale_ang=32
        scale_freq=64

        EPOCHS = 2000
        CAUS_CRIT = "slope"
        LOSS_TYPE = "atf"
        DO_CALIB = 0
        DO_BAR = 0
        N_MICS = 6
        template 'main_losses.sh'
}



DATA_H5 = Channel.fromPath("/home/dicarlod/Documents/Code/SteerAndInterp/data/easycom/Easycom_N-1020_fs-48k_nrfft-257.h5")
BASELINES = ["SCF", "SIREN"]
BASELINES1 = ["SCF1", "SIREN1"]
ARCHITECTURE = ["SIREN_PHASE", "SIREN_PHASE_CASCADE"]
// fARCHITECTURE = ["SIREN_PHASE", "SIREN_PHASE_CASCADE", "fSIREN_PHASE", "fSIREN_PHASE_CASCADE"]
N_MICS = [1, 6]
DO_PHYSICS = [0, 1]
DO_BAR = [0, 1]
DO_CALIB = [0, 1]
EPOCHS_RUN = 2000
CAUS_CRIT = ["hilbert"]
LOSS_TYPE = ["mag_phase", "mag_phase_air", "mag_phase_air_caus", "mag_phase_air_caus_svec"]
SR_DS = [15, 30, 45, 60, 75, 90]
HIDDEN_DIM = [512]
HIDDEN_NUM = [4]

// // HyperParameters for Dataset Creationg
// EASYCOM_H5 = Channel.fromPath("/media/dicarlod/SSD_2/diego/Dataset/SPEAR22/Miscellaneous/Array_Transfer_Functions/Device_ATFs.h5")
// N_RFFT = [257, 513, 1025]

workflow {
    main:
        // create_easycom_datasets(EASYCOM_H5, N_RFFT)
        run_baselines(DATA_H5, BASELINES1, N_MICS)
        best_simple_model_freqs_out(DATA_H5,  ARCHITECTURE,  2000, N_MICS, DO_PHYSICS, LOSS_TYPE, CAUS_CRIT, DO_BAR, DO_CALIB)
        best_simple_model_freqs_in( DATA_H5,  ARCHITECTURE,  2000, N_MICS, DO_PHYSICS, LOSS_TYPE, CAUS_CRIT, DO_BAR, DO_CALIB)
        super_resolution_random_grid_baselines( DATA_H5, BASELINES1, SR_DS)
        super_resolution_random_freqs_in(  DATA_H5,  ARCHITECTURE,  2000, SR_DS, HIDDEN_DIM, HIDDEN_NUM, LOSS_TYPE)
        super_resolution_random_freqs_out( DATA_H5,  ARCHITECTURE,  2000, SR_DS, HIDDEN_DIM, HIDDEN_NUM, LOSS_TYPE)
}