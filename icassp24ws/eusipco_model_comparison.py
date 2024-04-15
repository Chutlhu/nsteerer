import uuid
import optuna
import argparse

from eusipco_main import run
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str)
parser.add_argument("--freqs", type=str)
parser.add_argument("--epochs", type=int, default=2000)

# 1. Define an objective function to be minimize.
def objective(trial, args):

    args = args

    freqs = args.freqs
    if freqs == "in":
        batch_size = 2**14
        architecture = trial.suggest_categorical(
            "architecture_freqs_in", ["SIREN", "SIREN_PHASE", "SIREN_PHASE_CASCADE", "SIREN_MAG_CASCADE",
                             "fSIREN", "fSIREN_PHASE", "fSIREN_PHASE_CASCADE",
                             "MFN", "MFN_PHASE", "MFN_PHASE_CASCADE", "fMFN"]
        )
        feature_freq = trial.suggest_categorical('feature_freq_in', [32, 64, 128])
    elif freqs == "out":
        batch_size = 18
        architecture = trial.suggest_categorical(
            "architecture_freqs_out", ["SIREN", "SIREN_PHASE", "SIREN_PHASE_CASCADE", "SIREN_MAG_CASCADE",
                                       "MFN", "MFN_PHASE", "MFN_PHASE_CASCADE"]
        )
        feature_freq = trial.suggest_categorical('feature_freq_out', [0])
    else:
        raise ValueError(f"Freqs either in or out, found {freqs}")

    # 2. Suggest values of the hyperparameters using a trial object.

    learning_rate = trial.suggest_categorical('learning_rate', [1e-4, 1e-3, 1e-2])   
    feature_ang = trial.suggest_categorical('feature_ang',       [16, 32, 64])
    hidden_num = trial.suggest_categorical('hidden_num',            [1, 3, 5])
    hidden_dim = trial.suggest_categorical('hidden_dim',     [32, 256, 1024])
    loss_mag_lam   = trial.suggest_categorical("loss_mag_lam",   [0, 1., 10])
    loss_phase_lam = trial.suggest_categorical("loss_phase_lam", [0, 1., 10])
    loss_airs_lam  = trial.suggest_categorical("loss_airs_lam",  [0, 1., 10])
    loss_caus_lam  = trial.suggest_categorical("loss_caus_lam",  [0, 1., 10])

    uuid_tag = str(uuid.uuid1()).split("-")[0]
    model = "dnn"
    ds = 2
    grid_type = "regular"
    grid_pattern = "checkboard"
    exp_dir = f"exp/{args.name}/{architecture}_{grid_type}_{grid_pattern}_ds-{ds}_{uuid_tag}_{trial.number}"

    conf = {
        "epochs" : args.epochs,
        "data_h5" : './data/easycom/Easycom_N-1020_fs-48k_nrfft-257.h5',
        "seed" : 999, 
        "ds" : ds,
        "grid_type" : grid_type,
        "grid_pattern" : grid_pattern,
        "model" : model,
        "exp_dir" : exp_dir,
        "batch_size" : batch_size,
        "architecture" : architecture,
        "do_svect" : False,
        "do_bar" : False,
        "do_delay" : False,
        "do_freqs_in" : freqs == "in",
        "lr" : learning_rate,
        "hidden_dim" : hidden_dim,
        "hidden_num" : hidden_num,
        "scale_ang" : feature_ang,
        "scale_freq" : feature_freq,
        "loss_mag_lam" : loss_mag_lam,
        "loss_phase_lam" : loss_phase_lam,
        "loss_airs_lam" : loss_airs_lam,
        "loss_caus_lam" : loss_caus_lam,
    }

    pprint(conf)

    error = run(conf, opt_trial=trial)

    return error


if __name__  == "__main__":
    args = parser.parse_args()

    study = optuna.create_study(
            study_name=args.name,
            storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
            direction='minimize',
            sampler=optuna.samplers.BruteForceSampler(),
            load_if_exists=True,
            
        )

    study.optimize(lambda trial: objective(trial, args))