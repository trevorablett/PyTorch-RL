import multiprocessing

from torchvision import transforms


from utils.replay_memory import Memory
from utils.torch import *
from models.cnn_common import img_transform, imgnet_means, imgnet_stds
import math
import time


dtype = torch.float32

def collect_samples(pid, queue, env, policy, custom_reward,
                    mean_action, render, running_state, min_batch_size, aux_running_state):
    torch.randn(pid)
    log = dict()
    memory = Memory()
    num_steps = 0
    total_reward = 0
    min_reward = 1e6
    max_reward = -1e6
    total_c_reward = 0
    min_c_reward = 1e6
    max_c_reward = -1e6
    num_episodes = 0
    aux_state = None
    next_aux_state = None
    car_racing_env = 'CarRacing' in str(env)
    is_img_state = len(env.observation_space.shape) == 3
    if car_racing_env:
        memory = Memory(include_aux_state=True)
    if is_img_state:
        img_t = img_transform(imgnet_means, imgnet_stds)

    while num_steps < min_batch_size:
        state = env.reset()
        if car_racing_env:
            aux_state = np.array([np.linalg.norm(env.car.hull.linearVelocity)])
        if running_state is not None:
            state = running_state(state)
        if aux_state is not None and aux_running_state is not None:
            aux_state = aux_running_state(aux_state)
        reward_episode = 0

        for t in range(10000):
            if is_img_state:
                state_var = img_t(state).unsqueeze(0)
            else:
                state_var = tensor(state).unsqueeze(0)
            if aux_state is not None:
                aux_state_var = tensor(aux_state).view(1, -1).to(dtype)

            with torch.no_grad():
                if mean_action:
                    if aux_state is not None:
                        action = policy(state_var, aux_state_var)[0][0].numpy()
                    else:
                        action = policy(state_var)[0][0].numpy()
                else:
                    if aux_state is not None:
                        action = policy.select_action(state_var, aux_state_var)[0].numpy()
                    else:
                        action = policy.select_action(state_var)[0].numpy()
            action = int(action) if policy.is_disc_action else action.astype(np.float64)
            next_state, reward, done, _ = env.step(action)
            if car_racing_env:
                next_aux_state = np.array([np.linalg.norm(env.car.hull.linearVelocity)])
            reward_episode += reward
            if running_state is not None:
                next_state = running_state(next_state)
            if next_aux_state is not None and aux_running_state is not None:
                next_aux_state = aux_running_state(next_aux_state)

            if custom_reward is not None:
                if is_img_state:
                    reward = custom_reward(state, action, aux_state)
                else:
                    reward = custom_reward(state, action)
                    total_c_reward += reward
                    min_c_reward = min(min_c_reward, reward)
                    max_c_reward = max(max_c_reward, reward)

            mask = 0 if done else 1

            if is_img_state:
                mem_state = state_var.squeeze().numpy()
                mem_next_state = img_t(next_state).numpy()
            else:
                mem_state = state
                mem_next_state = next_state
            if aux_state is not None:
                memory.push(mem_state, action, mask, mem_next_state, reward, aux_state, next_aux_state)
            else:
                memory.push(mem_state, action, mask, mem_next_state, reward)

            if render:
                env.render()
            if done:
                break

            # temp for testing in car env
            if t > 100:
                break

            state = next_state
            if aux_state is not None:
                aux_state = next_aux_state

        # log stats
        num_steps += (t + 1)
        num_episodes += 1
        total_reward += reward_episode
        min_reward = min(min_reward, reward_episode)
        max_reward = max(max_reward, reward_episode)

    log['num_steps'] = num_steps
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = total_reward / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    if custom_reward is not None:
        log['total_c_reward'] = total_c_reward
        log['avg_c_reward'] = total_c_reward / num_steps
        log['max_c_reward'] = max_c_reward
        log['min_c_reward'] = min_c_reward

    if queue is not None:
        queue.put([pid, memory, log])
    else:
        return memory, log


def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])
    if 'total_c_reward' in log_list[0]:
        log['total_c_reward'] = sum([x['total_c_reward'] for x in log_list])
        log['avg_c_reward'] = log['total_c_reward'] / log['num_steps']
        log['max_c_reward'] = max([x['max_c_reward'] for x in log_list])
        log['min_c_reward'] = min([x['min_c_reward'] for x in log_list])

    return log


class Agent:

    def __init__(self, env, policy, device, custom_reward=None,
                 mean_action=False, render=False, running_state=None, num_threads=1,
                 aux_running_state=None):
        self.env = env
        self.policy = policy
        self.device = device
        self.custom_reward = custom_reward
        self.mean_action = mean_action
        self.running_state = running_state
        self.aux_running_state = aux_running_state
        self.render = render
        self.num_threads = num_threads

    def collect_samples(self, min_batch_size):
        t_start = time.time()
        to_device(torch.device('cpu'), self.policy)
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
        queue = multiprocessing.Queue()
        workers = []

        for i in range(self.num_threads-1):
            worker_args = (i+1, queue, self.env, self.policy, self.custom_reward, self.mean_action,
                           False, self.running_state, thread_batch_size, self.aux_running_state)
            workers.append(multiprocessing.Process(target=collect_samples, args=worker_args))
        for worker in workers:
            worker.start()

        memory, log = collect_samples(0, None, self.env, self.policy, self.custom_reward, self.mean_action,
                                      self.render, self.running_state, thread_batch_size, self.aux_running_state)

        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)
        for _ in workers:
            pid, worker_memory, worker_log = queue.get()
            worker_memories[pid - 1] = worker_memory
            worker_logs[pid - 1] = worker_log
        for worker_memory in worker_memories:
            memory.append(worker_memory)
        batch = memory.sample()
        if self.num_threads > 1:
            log_list = [log] + worker_logs
            log = merge_log(log_list)
        to_device(self.device, self.policy)
        t_end = time.time()
        log['sample_time'] = t_end - t_start
        log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
        log['action_min'] = np.min(np.vstack(batch.action), axis=0)
        log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        return batch, log
