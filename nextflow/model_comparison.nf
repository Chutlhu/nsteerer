
pyfile = file("./eusipco23/eusipco_main.py")
exp_name = "model_comparison"

process model_comparison_freqs_in {
    publishDir "./eusipco23/evaluation/$exp_name", pattern: "exp_*", mode: 'copy'
    input:
        path DATA
        each ARCHITECTURE
        val EPOCHS
        each N_MICS
        each DO_PHYSICS
        each LOSS_TYPE
        each CAUS_CRIT
    output:
        path("exp_*")
    script:
        freqs_in=1
        grid_type="regular"
        ds=2
        hidden_dim=64
        hidden_num=3
        lr=0.001
        template 'main_losses.sh'
}

process model_comparison_freqs_out {
    publishDir "./eusipco23/evaluation/$exp_name", pattern: "exp_*"
    input:
        path DATA
        each ARCHITECTURE
        val EPOCHS
        each N_MICS
        each DO_PHYSICS
        each LOSS_TYPE
        each CAUS_CRIT
    output:
        path("exp_*")
    script:
        freqs_in=0
        grid_type="regular"
        ds=2
        hidden_dim=256
        hidden_num=2
        lr=0.001
        template 'main_losses.sh'
}

DATA = Channel.fromPath("./data/easycom/Easycom_N-1020_fs-48k_nrfft-257.h5")
// ARCHITECTURE = ["SIREN_PHASE", "SIREN_PHASE_CASCADE", "SIREN_PHASE_CASCADE_MAG"]
// fARCHITECTURE = ["SIREN_PHASE", "SIREN_PHASE_CASCADE", "SIREN_PHASE_CASCADE_MAG", 
//                  "fSIREN", "fSIREN_PHASE", "fSIREN_PHASE_CASCADE"]
ARCHITECTURE = ["MFN", "MFN_PHASE", "MFN_PHASE_CASCADE"]
N_MICS = [1, 4]
DO_PHYSICS = [1]
// CAUS_CRIT = ["unwrap", "slope", "hilbert"]
CAUS_CRIT = ["slope"]
LOSS_TYPE = ["mag_phase", "mag_phase_air", "mag_phase_air_caus"]


workflow {
    main:
        // model_comparison_freqs_out(DATA,  ARCHITECTURE,  1, N_MICS, DO_PHYSICS, LOSS_TYPE, CAUS_CRIT)
        // model_comparison_freqs_in(DATA, fARCHITECTURE,  1, N_MICS, DO_PHYSICS, LOSS_TYPE, CAUS_CRIT)
        model_comparison_freqs_out(DATA, ARCHITECTURE,  2000, N_MICS, DO_PHYSICS, LOSS_TYPE, CAUS_CRIT)
        model_comparison_freqs_in( DATA, ARCHITECTURE, 2000, N_MICS, DO_PHYSICS, LOSS_TYPE, CAUS_CRIT)
}