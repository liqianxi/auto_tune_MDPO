#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI

from stable_baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import bench, logger
from stable_baselines.sac import SAC
import stable_baselines.common.tf_util as tf_util
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
import gym
import random
def train(env_id, num_timesteps, seed, sgd_steps, log):
    """
    Train TRPO model for the mujoco environment, for testing purposes
    :param env_id: (str) Environment ID
    :param num_timesteps: (int) The total number of samples
    :param seed: (int) The initial seed for training
    """
    with tf_util.single_threaded_session():
        rank = MPI.COMM_WORLD.Get_rank()
        seed = MPI.COMM_WORLD.Get_rank()
        log_path = './experiments/'+str(env_id)+'./SAC/bootstrap-2/m'+str(sgd_steps)+'_e'+'_'+str(seed)+"randomid"+str(random.random())
        if not log:
            #if rank == 0:
            logger.configure(log_path)
            #else:
            #    logger.configure(log_path, format_strs=[])
            #    logger.set_level(logger.DISABLED)
        else:
            if rank == 0:
                logger.configure()
            else:
                logger.configure(format_strs=[])
                logger.set_level(logger.DISABLED)
        
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()

        #env = make_mujoco_env(env_id, workerseed)
        def make_env():
            env_out = gym.make(env_id)
            env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
            env_out.seed(seed)
            return env_out

        env = DummyVecEnv([make_env])
        env = VecNormalize(env, norm_reward=False, norm_obs=False)
        
        #env = VecNormalize(env)
        model = SAC(MlpPolicy, env, gamma=0.99, verbose=1, gradient_steps=int(sgd_steps), \
             train_freq=2, learning_starts=10000,tau=0.005,learning_rate=3e-4,buffer_size=1e6 ,tensorboard_log="./tensorboard_log")

        model.learn(total_timesteps=int(num_timesteps))
        env.close()


def main():
    """
    Runs the test
    """
    args = mujoco_arg_parser().parse_args()
    print(args)
    train(args.env, num_timesteps=args.num_timesteps, seed=args.run, sgd_steps=args.sgd_steps, log=args.log)


if __name__ == '__main__':
    main()
