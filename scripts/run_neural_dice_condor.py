import argparse
import itertools
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", help="condor cluster id", default=0, type=int)
    parser.add_argument("--pid", help="condor process id", default=0, type=int)
    args = parser.parse_args()

    ## ==========================
    ## Experiment Hyperparameters
    ## ==========================
    
    PYTHON_BIN          = "/home/twguntara/miniconda3/envs/dice-rl/bin/python"
    PROJECT_DIR         = "/ext2/twguntara/dice_rl"

    seed_list   = [i for i in range(10)]
    reg_list    = [
        # (0., 1., 1, 0., 0, "DualDICE"),
        # (1., 0., 1, 1., 1, "GenDICE"),
        # (1., 0., 1, 1., 0, "GradientDICE"),
        (0., 1., 0, 1., 1, "BestDICE")
    ]
    env_name    = "Pendulum-v1"
    buffer_path = "scripts/dataset/Pendulum-v1/tf_buffer/medium_500000_dummy1_rand0.20_bias0.00_std0.50"
    policy_path = "scripts/dataset/Pendulum-v1/policies/expert/actor"
    save_path   = "/tmp/twguntara"
    gamma       = 0.95
    unique_id   = "condor"
    num_steps   = int(2e6)
    batch_size  = 1024

    combinations    = list(itertools.product(seed_list, reg_list))

    # Jobs configuration
    cmds = []
    for hparams in combinations:
        seed                = hparams[0]
        primal_regularizer  = hparams[1][0]
        dual_regularizer    = hparams[1][1]
        zero_reward         = hparams[1][2]
        norm_regularizer    = hparams[1][3]
        zeta_pos            = hparams[1][4]
        algo_name           = hparams[1][5]

        cmd = "%s scripts/run_neural_dice.py" % PYTHON_BIN
        cmd += " --save_dir %s" % save_path
        cmd += " --policy_filepath %s" % policy_path
        cmd += " --buffer_filepath %s" % buffer_path
        cmd += " --env_name %s" % env_name
        cmd += " --seed %d" % seed
        cmd += " --primal_regularizer %f" % primal_regularizer
        cmd += " --dual_regularizer %f" % dual_regularizer
        cmd += " --zero_reward %d" % zero_reward
        cmd += " --norm_regularizer %f" % norm_regularizer
        cmd += " --zeta_pos %d" % zeta_pos
        cmd += " --algo_name %s" % algo_name
        cmd += " --gamma %f" % gamma
        cmd += " --num_steps %d" % num_steps
        cmd += " --batch_size %d" % batch_size
        cmd += " --unique_id %s" % unique_id

        cmds.append(cmd)

    for pid, cmd in enumerate(cmds):
        print(f'[{pid}] {cmd}', flush=True)
    print('==================', flush=True)
    print(args.pid, cmds[args.pid], flush=True)
    print('==================', flush=True)

    if args.cid == 0:
        print('Run:', flush=True)
        print(f'condor_submit {PROJECT_DIR}/condor_script/condor.submit -queue {len(cmds)}', flush=True)
    else:
        print("Start running", flush=True)
        cmd = cmds[args.pid]
        os.system(f"cd {PROJECT_DIR}; " + cmd)