import argparse
import gym
import os
import sys
import pickle
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy import Policy
from models.cnn_policy import CNNPolicy
from models.mlp_critic import Value
from models.cnn_critic import CNNValue
from models.mlp_policy_disc import DiscretePolicy
from models.mlp_discriminator import Discriminator
from models.cnn_discriminator import CNNDiscriminator
from torch import nn
from torchvision import transforms
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent
from models.cnn_common import img_transform, imgnet_means, imgnet_stds


parser = argparse.ArgumentParser(description='PyTorch GAIL example')
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--expert-traj-path', default="../assets/expert_traj/Hopper-v2_expert_traj.p",
                    metavar='G', help='path of the expert trajectories')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=-1.0, metavar='G',
                    help='log std for the policy (default: -1.0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='gae (default: 3e-4)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
parser.add_argument('--pre-train', action='store_true', default=False,
                    help='pre-train the policy using behavioral cloning')
args = parser.parse_args()

dtype = torch.float32
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)

"""CNN options (hardcoded for now for simplicity)"""
cnn_options = dict(
    channels = [64, 32, 32],
    kernel_sizes = [7, 4, 3],
    strides = [4, 2, 1],
    head_hidden_sizes = [512]
)

img_t = img_transform(imgnet_means, imgnet_stds)

"""environment"""
env = gym.make(args.env_name)
# todo no policy for image state and discrete action
is_img_state = len(env.observation_space.shape) == 3
if not is_img_state:
    state_dim = env.observation_space.shape[0]
    running_state = ZFilter((state_dim,), clip=5)
else:
    state_dim = env.observation_space.shape
    running_state = ZFilter(state_dim, clip=5)
is_disc_action = len(env.action_space.shape) == 0
action_dim = 1 if is_disc_action else env.action_space.shape[0]
# running_reward = ZFilter((1,), demean=False, clip=10)

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

"""define actor and critic"""
if args.env_name == 'CarRacing-v0':
    num_aux = 1
else:
    num_aux = 0

if num_aux > 0:
    aux_running_state = ZFilter(num_aux, clip=5)
else:
    aux_running_state = None

if is_disc_action:
    policy_net = DiscretePolicy(state_dim, env.action_space.n)
elif is_img_state:
    policy_net = CNNPolicy(state_dim, env.action_space.shape[0], cnn_options['channels'],
                           cnn_options['kernel_sizes'], cnn_options['strides'],
                           head_hidden_size=cnn_options['head_hidden_sizes'],
                           num_aux=num_aux, log_std=args.log_std)
else:
    policy_net = Policy(state_dim, env.action_space.shape[0], log_std=args.log_std)

if is_img_state:
    value_net = CNNValue(state_dim, cnn_options['channels'],
                           cnn_options['kernel_sizes'], cnn_options['strides'],
                           head_hidden_size=cnn_options['head_hidden_sizes'],
                           num_aux=num_aux)
    discrim_net = CNNDiscriminator(state_dim, action_dim, cnn_options['channels'],
                           cnn_options['kernel_sizes'], cnn_options['strides'],
                           head_hidden_size=cnn_options['head_hidden_sizes'],
                           num_aux=num_aux)
else:
    value_net = Value(state_dim)
    discrim_net = Discriminator(state_dim + action_dim)

discrim_criterion = nn.BCELoss()
to_device(device, policy_net, value_net, discrim_net, discrim_criterion)

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)
optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=args.learning_rate)

# optimization epoch number and batch size for PPO
optim_epochs = 10
optim_batch_size = 64

# load trajectory
if is_img_state:
    expert_data = pickle.load(open(args.expert_traj_path, "rb"))

    # TODO remove these 3 lines... for now, just copying dict keys to generic names that will be used for new datasets
    expert_data['states'] = expert_data['obs']
    expert_data['actions'] = expert_data['act']
    expert_data['aux'] = expert_data['vel']

    if num_aux > 0:
        aux_running_state.rs._M = np.atleast_1d(expert_data['aux'].mean())
        aux_running_state.rs._n = np.atleast_1d(expert_data['aux'].shape[0])
        # see https://www.johndcook.com/blog/standard_deviation/ for calc below
        aux_running_state.rs._S = expert_data['aux'].var() * (aux_running_state.rs.n - 1)
        expert_data['aux'] = aux_running_state(expert_data['aux'], update=False)

    # convert all expert data to torch tensors
    # for loop might be slow, should be parallelized with dataloader or just multiprocessing
    transformed_imgs = []
    for i in range(expert_data['states'].shape[0]):
        transformed_imgs.append(img_t(expert_data['states'][i]))
        if i % 1000 == 0:
            print("Transforming expert img %d of %d..." % (i, expert_data['states'].shape[0]))
    expert_data['states'] = torch.stack(transformed_imgs).to(dtype).to(device)
    expert_data['actions'] = torch.tensor(expert_data['actions']).to(dtype).to(device)
    if num_aux > 0:
        expert_data['aux'] = torch.tensor(expert_data['aux']).to(dtype).to(device)
        if num_aux == 1:
            expert_data['aux'] = expert_data['aux'].unsqueeze(1)

    running_state = None
    num_expert_data = expert_data['states'].shape[0]
else:
    expert_traj, running_state = pickle.load(open(args.expert_traj_path, "rb"))
    num_expert_data = expert_traj.shape[0]
# expert_data = pickle.load(open(args.expert_traj_path, "rb"))

# running_state is normally just a z-filter for updating means and std as well as normalization,
# as an alternative, just normalize states before passing through models

# TODO next step: ensure that data from expert trajs matches what the expert traj here looks like,
# or figure out a way to make it work nicely

# TODO need to add in option for pretraining / BC

# TODO multiprocessing doesn't work properly for carracing (fails after one iteration)


"""optional pretraining"""
if args.pre_train:
    # params could be user set, but basic params should be fine for most cases
    max_pt_epochs = 500
    max_pt_epochs_wo_best = 30
    pt_lr = 1e-3
    # could also be done with log-likelihood, see
    # https://github.com/aravindr93/mjrl/blob/master/mjrl/algos/behavior_cloning.py
    pt_criterion = nn.MSELoss()

    # todo ensure that the std output from the policy net is still reasonable after

    perm = np.arange(num_expert_data)
    np.random.shuffle(perm)
    perm = LongTensor(perm).to(device)
    valid_perm = perm[:math.floor(.2 * num_expert_data)]
    train_perm = perm[math.floor(.2 * num_expert_data):]
    lowest_valid_loss = 1e10
    epochs_wo_best = 0
    # for i in range()

    if is_img_state:
        valid_perm = expert_data['states']num_expert_data * .2


def expert_reward(state, action, aux_state=None):
    is_img_state = len(env.observation_space.shape) == 3  # shadows outer scope var in case this func is moved
    if is_img_state:
        state_var = img_t(state).unsqueeze(0)
        if aux_state is not None:
            aux_state_var = tensor(aux_state, dtype=dtype).unsqueeze(0)
        else:
            aux_state_var = None
        action_var = tensor(action, dtype=dtype).unsqueeze(0)
        return -math.log(discrim_net(state_var, action_var, aux_state_var))
    else:
        state_action = tensor(np.hstack([state, action]), dtype=dtype)
        with torch.no_grad():
            return -math.log(discrim_net(state_action)[0].item())


"""create agent"""
agent = Agent(env, policy_net, device, custom_reward=expert_reward,
              running_state=running_state, render=args.render, num_threads=args.num_threads,
              aux_running_state=aux_running_state)


def update_params(batch, i_iter):
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    if type(batch).__name__ == 'TransitionWithAux':
        aux_states = torch.from_numpy(np.stack(batch.aux_state)).to(dtype).to(device)
    else:
        aux_states = None
    with torch.no_grad():
        if aux_states is not None:
            values = value_net(states, aux_states)
            fixed_log_probs = policy_net.get_log_prob(states, actions, aux_states)  # sums log probs for all actions
        else:
            values = value_net(states)
            fixed_log_probs = policy_net.get_log_prob(states, actions)

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)

    """update discriminator"""
    for _ in range(1):
        if is_img_state:
            g_o = discrim_net(states, actions, aux_states)
            e_o = discrim_net(expert_data['states'], expert_data['actions'], expert_data['aux'])
        else:
            expert_state_actions = torch.from_numpy(expert_traj).to(dtype).to(device)
            g_o = discrim_net(torch.cat([states, actions], 1))
            e_o = discrim_net(expert_state_actions)
        optimizer_discrim.zero_grad()
        discrim_loss = discrim_criterion(g_o, ones((states.shape[0], 1), device=device)) + \
            discrim_criterion(e_o, zeros((num_expert_data, 1), device=device))
        discrim_loss.backward()
        optimizer_discrim.step()

    """perform mini-batch PPO update"""
    optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = LongTensor(perm).to(device)

        states, actions, returns, advantages, fixed_log_probs = \
            states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), fixed_log_probs[perm].clone()
        if aux_states is not None:
            aux_states = aux_states[perm].clone()

        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
            states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]
            if aux_states is not None:
                aux_states_b = aux_states[ind]
            else:
                aux_states_b = None

            ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 1, states_b, actions_b, returns_b,
                     advantages_b, fixed_log_probs_b, args.clip_epsilon, args.l2_reg, aux_states_b)


def main_loop():
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        discrim_net.to(torch.device('cpu'))
        batch, log = agent.collect_samples(args.min_batch_size)
        discrim_net.to(device)

        t0 = time.time()
        update_params(batch, i_iter)
        t1 = time.time()

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\texpert_R_avg {:.2f}\tR_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, log['avg_c_reward'], log['avg_reward']))

        if args.save_model_interval > 0 and (i_iter+1) % args.save_model_interval == 0:
            to_device(torch.device('cpu'), policy_net, value_net, discrim_net)
            pickle.dump((policy_net, value_net, discrim_net), open(os.path.join(assets_dir(), 'learned_models/{}_gail.p'.format(args.env_name)), 'wb'))
            to_device(device, policy_net, value_net, discrim_net)

        """clean up gpu memory"""
        torch.cuda.empty_cache()


main_loop()
