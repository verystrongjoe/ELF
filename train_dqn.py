"""
--batchsize 128
--freq_update 1
--players "type=AI_NN,fs=50,args=backup/AI_SIMPLE|start/500|decay/0.99;type=AI_SIMPLE,fs=20"
--num_games 1024
--tqdm
--T 20
--additional_labels id,last_terminal
--trainer_stats winrate
--keys_in_reply V
--gpu 0
"""
from datetime import datetime
import sys
import os

from rlpytorch import *

if __name__ == '__main__':

    os.environ['game'] = './rts/game_MC/game'
    os.environ['model_file'] = './rts/game_MC/model'
    os.environ['model'] = 'dqn'

    trainer = Q_Trainer()
    runner = SingleProcessRun()

    env, all_args = load_env(os.environ, trainer=trainer, runner=runner)
    GC = env["game"].initialize()

    model = env["model_loaders"][0].load_model(GC.params)

    env["mi"].add_model("model", model, opt=True)
    env["mi"].add_model("actor", model, copy=True, cuda=all_args.gpu is not None, gpu_id=all_args.gpu)

    env["sampler"].args.sample_policy = 'q_learning_epsilon_greedy'

    trainer.setup(mi=env["mi"], rl_method=env["method"], sampler=env["sampler"])

    GC.reg_callback("train", trainer.train)
    GC.reg_callback("actor", trainer.actor)
    runner.setup(GC, episode_summary=trainer.episode_summary, episode_start=trainer.episode_start)
    runner.run()

