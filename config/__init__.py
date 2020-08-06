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
    parser.add_argument("--max_episode_steps", type=str, default=200)

    # training algorithm
    parser.add_argument("--algo", type=str, default="sac",
                        choices=["sac"])
    parser.add_argument("--policy", type=str, default="mlp",
                        choices=["mlp"])

    # vanilla rl
    parser.add_argument("--rl_hid_size", type=int, default=256)
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "elu", "tanh"])
    parser.add_argument("--tanh_policy", type=str2bool, default=True)
    parser.add_argument("--actor_num_hid_layers", type=int, default=2)

    # off-policy rl
    parser.add_argument("--buffer_size", type=int, default=int(1e3),
                        help="the size of the buffer (# episodes)")
    parser.add_argument("--discount_factor", type=float, default=0.99,
                        help="the discount factor")
    parser.add_argument("--lr_actor", type=float, default=3e-4,
                        help="the learning rate of the actor")
    parser.add_argument("--lr_critic", type=float, default=3e-4,
                        help="the learning rate of the critic")
    parser.add_argument("--polyak", type=float, default=0.995,
                        help="the average coefficient")
    parser.add_argument("--start_steps", type=float, default=10000)

    # training
    parser.add_argument("--is_train", type=str2bool, default=True)
    parser.add_argument("--num_batches", type=int, default=1,
                        help="the times to update the network per epoch")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="the sample batch size")
    parser.add_argument("--max_grad_norm", type=float, default=100)
    parser.add_argument("--max_global_step", type=int, default=int(2e6))
    parser.add_argument("--gpu", type=int, default=None)

    # sac
    parser.add_argument("--reward_scale", type=float, default=1.0, help="reward scale")
    parser.add_argument("--alpha", type=float, default=1.0)

    # ppo
    parser.add_argument("--clip_param", type=float, default=0.2)
    parser.add_argument("--value_loss_coeff", type=float, default=0.5)
    parser.add_argument("--action_loss_coeff", type=float, default=1.0)
    parser.add_argument("--entropy_loss_coeff", type=float, default=1e-4)
    parser.add_argument("--rollout_length", type=int, default=1000)
    parser.add_argument("--gae_lambda", type=float, default=0.95)

    # log
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--evaluate_interval", type=int, default=1000)
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
