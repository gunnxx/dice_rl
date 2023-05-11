import itertools
import multiprocessing
import os
import signal
import subprocess
import sys
import time

MAX_JOB      = 4
START_JOB_ID = 0
NUM_QUEUE    = 8
NUM_GPU      = 4

def kill_jobs():
    for job_id in running_job_ids:
        proc, f_out, f_err = job_id_to_proc[job_id]
        pid = proc.pid
        os.system('kill -9 %d' % pid)
        f_out.close()
        f_err.close()
        print('- Job %d is killed.' % job_id)

def current_jobs():
    result = []
    for job_id in running_job_ids:
        proc, f_out, f_err = job_id_to_proc[job_id]
        if proc.poll() is None:
            result.append(job_id)
        else:
            f_out.close()
            f_err.close()
    return result

def print_jobs_status():
    print('Running jobs: %s' % ','.join([str(x) for x in running_job_ids]))

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    kill_jobs()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

running_job_ids = []
job_id_to_proc = {}

job_id = START_JOB_ID

###############################################################################################################

seed_list   = [i for i in range(8)]
reg_list    = [
    # (0., 1., 1, 0., 0, "DualDICE"),
    # (1., 0., 1, 1., 1, "GenDICE"),
    # (1., 0., 1, 1., 0, "GradientDICE"),
    (0., 1., 0, 1., 1, "BestDICE")
]
# env_name    = "maze2d-umaze-v0"
# buffer_path = "scripts/dataset/maze2d-umaze-v0/tf_buffer/medium_60000_rand0.2_bias0.0_std0.2"
# policy_path = "scripts/dataset/maze2d-umaze-v0/policies/expert/actor"
env_name    = "Pendulum-v1"
buffer_path = "scripts/dataset/Pendulum-v1/tf_buffer/medium_500000_dummy1_rand0.20_bias0.00_std0.50"
policy_path = "scripts/dataset/Pendulum-v1/policies/expert/actor"
save_path   = "/ext_hdd/twguntara/kernel_ope"
gamma       = 0.95

combinations    = list(itertools.product(seed_list, reg_list))

###############################################################################################################

os.makedirs('%s/job_logs' % save_path, exist_ok=True)
while True:
    running_jobs = current_jobs()
    for running_job_id in running_job_ids:
        if not(running_job_id in running_jobs):
            running_job_ids.remove(running_job_id)
            del job_id_to_proc[running_job_id]
            print('- Job %d is finished.' % running_job_id)
            print_jobs_status()

    if len(running_jobs) >= MAX_JOB:
        time.sleep(1)
        continue

    if job_id >= START_JOB_ID + NUM_QUEUE:
        if len(current_jobs()) == 0:
            break
        time.sleep(1)
        continue

    ###############################################################################################################

    config = combinations[job_id]

    seed                = config[0]
    primal_regularizer  = config[1][0]
    dual_regularizer    = config[1][1]
    zero_reward         = config[1][2]
    norm_regularizer    = config[1][3]
    zeta_pos            = config[1][4]
    algo_name           = config[1][5]

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cmd = "exec python scripts/run_neural_dice.py"
    cmd += " --save_dir %s" % save_path
    cmd += " --policy_filepath %s" % policy_path
    cmd += " --buffer_filepath %s" % buffer_path
    cmd += " --env_name %s" % env_name
    cmd += " --seed=%d" % seed
    cmd += " --primal_regularizer=%f" % primal_regularizer
    cmd += " --dual_regularizer=%f" % dual_regularizer
    cmd += " --zero_reward=%d" % zero_reward
    cmd += " --norm_regularizer=%f" % norm_regularizer
    cmd += " --zeta_pos=%d" % zeta_pos
    cmd += " --algo_name=%s" % algo_name
    cmd += " --gamma %f" % gamma
    cmd += " --num_steps 2000000"
    cmd += " --batch_size 1024"

    ###############################################################################################################

    f_out = open('%s/job_logs/%d.out' % (save_path, job_id), 'w')
    f_err = open('%s/job_logs/%d.err' % (save_path, job_id), 'w')
    proc = subprocess.Popen(cmd, shell=True, stdout=f_out, stderr=f_err)
    job_id_to_proc[job_id] = (proc, f_out, f_err)

    running_job_ids.append(job_id)
    print("+ Job %d is submitted." % job_id)
    job_id += 1
    print_jobs_status()

print("FINISHED!")