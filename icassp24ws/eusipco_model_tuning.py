import uuid
import optuna
import argparse

from eusipco_main import run
from pprint import pprint

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str)
parser.add_argument("--db_path", type=str)
parser.add_argument("--tag", type=str)
parser.add_argument("--data_h5", type=str)
parser.add_argument("--n_trials", type=int, default=8)
parser.add_argument("--freqs", type=str, default='in')
parser.add_argument("--model", type=str, default='dnn')
parser.add_argument("--n_mics", type=int, default=1)
parser.add_argument("--architecture", type=str, default='SIREN')
parser.add_argument("--epochs", type=int, default=2000)
parser.add_argument("--ds", type=int, default=1)
parser.add_argument("--grid_type", type=str, default="regular")
parser.add_argument('--svect', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--delay', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--calib', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--baryc', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--metric', type=str, default="rmse_time")


# 1. Define an objective function to be minimize.
def objective(trial, args):

    args = args
    freqs = args.freqs

    learning_rate = trial.suggest_categorical('learning_rate', [1e-4, 1e-3, 1e-2])   
    batch_size = trial.suggest_int('batch_size', 16, 512, log=True)   
    scale_ang = trial.suggest_int('scale_ang',   16, 256, log=True)
    scale_freq = trial.suggest_int('scale_freq', 16, 256, log=True)
    hidden_num = trial.suggest_int('hidden_num', 1, 5)
    hidden_dim = trial.suggest_int('hidden_dim', 32, 1024, log=True)
    loss_mag_lam =   trial.suggest_categorical("loss_mag_lam",   [0, 0.1, 1, 10])
    loss_phase_lam = trial.suggest_categorical("loss_phase_lam", [0, 0.1, 1, 10])
    loss_air_lam =   trial.suggest_categorical("loss_air_lam",   [0, 0.1, 1, 10])
    loss_caus_lam =  trial.suggest_categorical("loss_caus_lam",  [0, 0.1, 1, 10])
    loss_caus_crit = trial.suggest_categorical("loss_caus_crit", ["hilbert", "slope", "unwrap"])

    do_svect = args.svect
    do_delay = args.delay
    do_calib = args.calib
    do_baryc = args.baryc

    model = args.model
    ds = args.ds
    grid_type = args.grid_type
    grid_pattern = "checkboard"
    exp_dir = f"evaluation/{args.name}/exp_{args.tag}_trial-{trial.number}/"
    
    conf = {
        "epochs" : args.epochs,
        "n_mics" : args.n_mics,
        "data_h5" : args.data_h5,
        "seed" : 999, 
        "ds" : ds,
        "grid_type" : grid_type,
        "grid_pattern" : grid_pattern,
        "model" : model,
        "exp_dir" : exp_dir,
        "batch_size" : batch_size,
        "architecture" : args.architecture,
        "monitor_metric" : args.metric,
        "do_svect" : do_svect,
        "do_bar"   : do_baryc,
        "do_delay" : do_delay,
        "do_freqs_in" : freqs == "in",
        "do_calib" : do_calib,
        "lars" : 0,
        "lr" : learning_rate,
        "hidden_dim" : hidden_dim,
        "hidden_num" : hidden_num,
        "scale_ang" : scale_ang,
        "scale_freq" : scale_freq,
        "loss_atf_lam" :   0,
        "loss_mag_lam" :   loss_mag_lam,
        "loss_phase_lam" : loss_phase_lam,
        "loss_air_lam" :   loss_air_lam,
        "loss_caus_lam" :  loss_caus_lam,
        "loss_caus_crit" : loss_caus_crit,
    }
    pprint(conf)

    metrics = run(conf, opt_trial=trial)
    return metrics["diff"][args.metric]


if __name__  == "__main__":
    args = parser.parse_args()

    study = optuna.create_study(
            study_name=args.name,
            storage=f"sqlite:///{args.db_path}",  # Specify the storage URL here.
            direction='minimize',
            sampler=optuna.samplers.RandomSampler(), # Bayesian search
            pruner=optuna.pruners.MedianPruner(), # Pruning        
            load_if_exists=True,
        )

    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)