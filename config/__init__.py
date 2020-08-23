import argparse

from util import str2bool, str2list


def argparser():
    parser = argparse.ArgumentParser(
        "Skill Coordination",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # environment
    parser.add_argument("--env", type=str, default="Reacher-v2",
                        help="Environment name")
    parser.add_argument("--task_name", type=str, default='run')
    parser.add_argument("--max_episode_steps", type=str, default=200)
    parser.add_argument("--img_height", type=int, default=84)
    parser.add_argument("--img_width", type=int, default=84)
    parser.add_argument("--pre_transform_image_size", type=int, default=100)
    parser.add_argument('--frame_stack', type=int, default=None)
    parser.add_argument("--action_repeat", type=int, default=2)

    # training algorithm
    parser.add_argument("--algo", type=str, default="sac",
                        choices=["sac", "td3", "ddpg", "ppo", "dqn"])
    parser.add_argument("--policy", type=str, default="mlp",
                        choices=["mlp", "cnn"])
    parser.add_argument("--unsup_algo", type=str, default='curl',
                        choices=['curl'])

    # vanilla rl
    parser.add_argument("--rl_hid_size", type=int, default=256)
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "elu", "tanh"])
    parser.add_argument("--tanh_policy", type=str2bool, default=True)
    parser.add_argument("--actor_num_hid_layers", type=int, default=2)
    parser.add_argument("--kernel_size", nargs='+', default=[3, 3, 3])
    parser.add_argument("--conv_dim", nargs='+', default=[32, 32, 32])
    parser.add_argument("--stride", nargs='+', default=[2, 1, 1])
    parser.add_argument("--encoder_feature_dim", type=int, default=50)

    # off-policy rl
    parser.add_argument("--buffer_size", type=int, default=int(1e6),
                        help="the size of the buffer (# episodes)")
    parser.add_argument("--discount_factor", type=float, default=0.99,
                        help="the discount factor")
    parser.add_argument("--lr_actor", type=float, default=3e-4,
                        help="the learning rate of the actor")
    parser.add_argument("--lr_critic", type=float, default=3e-4,
                        help="the learning rate of the critic")
    parser.add_argument("--lr_encoder", type=float, default=1e-3,
                        help="the learning rate of the encoder")
    parser.add_argument("--polyak", type=float, default=0.995,
                        help="the average coefficient")
    parser.add_argument("--init_steps", type=float, default=10000)

    # training
    parser.add_argument("--is_train", type=str2bool, default=True)
    parser.add_argument("--num_batches", type=int, default=1,
                        help="the times to update the network per epoch")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="the sample batch size")
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--max_global_step", type=int, default=int(2e6))
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--multiprocessing", type=str2bool, default=False)
    parser.add_argument("--num_processes", type=int, default=16)

    # sac
    parser.add_argument("--reward_scale", type=float, default=1.0, help="reward scale")
    parser.add_argument("--alpha", type=float, default=1.0)

    # td3
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--noise_clip", type=float, default=0.5)

    # ppo
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--value_loss_coeff", type=float, default=0.5)
    parser.add_argument("--action_loss_coeff", type=float, default=1.0)
    parser.add_argument("--entropy_loss_coeff", type=float, default=1e-4)
    parser.add_argument("--rollout_length", type=int, default=1000)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ppo_epoch", type=int, default=10)

    # log
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--evaluate_interval", type=int, default=100)
    parser.add_argument("--ckpt_interval", type=int, default=1e5)
    parser.add_argument("--log_root_dir", type=str, default="logs")
    parser.add_argument('--wandb', type=str2bool, default=False,
                        help="set it True if you want to use wandb")
    parser.add_argument("--entity", type=str, default="")
    parser.add_argument("--project", type=str, default="")

    # evaluation
    parser.add_argument("--ckpt_num", type=int, default=None)
    parser.add_argument("--num_eval", type=int, default=10)
    parser.add_argument("--save_rollout", type=str2bool, default=False,
                        help="save rollout information during evaluation")
    parser.add_argument("--record", type=str2bool, default=True)
    parser.add_argument("--record_caption", type=str2bool, default=True)
    parser.add_argument("--num_record_samples", type=int, default=1,
                        help="number of trajectories to collect during eval")

    # misc
    parser.add_argument("--prefix", type=str, default="test")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--virtual_display", type=str, default=":1",
                        help="Specify virtual display for rendering if you use (e.g. ':0' or ':1')")

    return parser
