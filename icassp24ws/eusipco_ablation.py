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

    if args.freqs == "in":
        batch_size = 2**14
    elif args.freqs == "out":
        batch_size = 18
    else:
        raise ValueError(f"Freqs either in or out, found {args.freqs}")

    # 2. Suggest values of the hyperparameters using a trial object.
    learning_rate = trial.suggest_categorical('learning_rate', [1e-5, 1e-4, 1e-3, 1e-2])   
    feature_ang = trial.suggest_int('feature_ang', 2, 512, log=True)
    hidden_num = trial.suggest_int('hidden_num', 1, 5)
    hidden_dim = trial.suggest_int('hidden_dim', 32, 1024, log=True)
    loss_mag_lam   = trial.suggest_categorical("loss_mag_lam",   [0, 0.01, 0.1, 1., 10])
    loss_phase_lam = trial.suggest_categorical("loss_phase_lam", [0, 0.01, 0.1, 1., 10])
    loss_airs_lam  = trial.suggest_categorical("loss_airs_lam",  [0, 0.01, 0.1, 1., 10])
    loss_caus_lam  = trial.suggest_categorical("loss_caus_lam",  [0, 0.01, 0.1, 1., 10])

    uuid_tag = str(uuid.uuid1()).split("-")[0]
    architecture = "SIREN"
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
        "do_freqs_in" : args.freqs == "in",
        "lr" : learning_rate,
        "hidden_dim" : hidden_dim,
        "hidden_num" : hidden_num,
        "scale_ang" : feature_ang,
        "scale_freq" : 0.,
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
    # 3. Create a study object and optimize the objective function.
    study = optuna.create_study(
            study_name=args.name,
            storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
            direction='minimize',
            sampler=optuna.samplers.RandomSampler(), # Bayesian search
            pruner=optuna.pruners.MedianPruner(), # Pruning        
            load_if_exists=True,  
        )
    study.optimize(lambda trial: objective(trial, args), n_trials=50)