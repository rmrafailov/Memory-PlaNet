import numpy as np
import torch
import cv2
import gym

def quantise_centre_dequantise(images, bit_depth):
    images.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)
    images.add_(torch.rand_like(images).div_(2 ** bit_depth))

def _images_to_observation(images, bit_depth):
    images = torch.tensor(cv2.resize(images, (64, 64), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32)
    quantise_centre_dequantise(images, bit_depth)
    return images.unsqueeze(dim=0)

class POPacman:
    colors = {"pacman": (210, 164, 74), # orange
              "black":(0, 0, 0),
              "wall":(228, 111, 111),
              "background": (0, 28, 136)}

    def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth, recentering=False):
        self._radius = 20
        self._recentering = recentering
        self._env = gym.make('MsPacman-v0')
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

        self.symbolic = symbolic
        self._env.seed(seed)
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.bit_depth = bit_depth

    def reset(self):
        self.t = 0
        state = self._env.reset()
        if self.symbolic:
            return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
        else:
            return _images_to_observation(self._env.render(mode='rgb_array'), self.bit_depth)

    def render(self):
        self._env.render()

    def render_partial(self, obs):
        im = plt.imshow(obs, animated=True)

    @property
    def observation_size(self):
        return self._env.observation_space.shape[0] if self.symbolic else (3, 64, 64)

    @property
    def action_size(self):
        sz = 1
        if not self._env.action_space.shape is ():
            sz = self._env.action_space.shape[0]
        return sz

    def sample_random_action(self):
        arr = np.array(self._env.action_space.sample())
        return torch.from_numpy(arr)

    def close(self):
        self._env.close()

    def step(self, action):
        if type(action) not in [float, int]:
            action = action.detach().numpy()
        reward = 0
        for k in range(self.action_repeat):
            state, reward_k, done, _ = self._env.step(action)
            reward += reward_k
            self.t += 1
            done = done or self.t == self.max_episode_length
            if done:
                break
        if self.symbolic:
            observation = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
        else:
            observation = self._env.render(mode='rgb_array')
            observation = self._adapt(observation)
            observation = _images_to_observation(observation, self.bit_depth)
        return observation, reward, done

    def _adapt(self, obs):
        radius = self._radius
        windows = (210, 160)
        board = (170, 160)

        indices = np.where(np.all(obs == self.colors["pacman"], axis=-1))
        mean_x, mean_y = np.mean(indices[0]), np.mean(indices[1])

        # Make circle
        xx, yy = np.mgrid[:windows[0], :windows[1]]
        circle = (xx-mean_x)**2+(yy-mean_y)**2
        mask = (circle < radius**2)
        mask = np.stack([mask]*3, axis=-1)
        new_obs = mask*obs

        if self._recentering:
            new_board = np.zeros((2*radius, 2*radius, 3), dtype=obs.dtype)
            max_x = min(int(mean_x + radius), windows[0])
            min_x = max(int(mean_x - radius), 0)
            lx = (max_x - min_x) // 2
            rx = (max_x - min_x) - lx
            max_y = min(int(mean_y + radius), windows[1])
            min_y = max(int(mean_y - radius), 0)
            ly = (max_y - min_y) // 2
            ry = (max_y - min_y) - ly
            new_board[radius-lx:radius+rx,
                      radius-ly:radius+ry] \
                = new_obs[min_x:max_x,
                          min_y:max_y]
            new_obs = new_board

        return new_obs
