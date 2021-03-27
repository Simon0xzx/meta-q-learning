import numpy as np
import metaworld
from . import register_env
@register_env('metaworld-ml1-window-close-v1')
class MetaworldML1WindowCloseEnv(metaworld.ML1):
    def __init__(self, n_tasks=50, randomize_tasks=True):
        super(MetaworldML1WindowCloseEnv, self).__init__('window-close-v1')
        self._env = self.train_classes['window-close-v1']()
        self.tasks = self.train_tasks
        self.reset_task(0)
        print("Init metaworld env")

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def sample_tasks(self, num_tasks):
        return np.random.choice(np.arange(50), num_tasks)

    def step(self, action):
        return self._env.step(action)

    def seed(self, seed):
        return self._env.seed(seed)

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx
        self._env.set_task(self._task)
        self._env.reset()

    def set_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx
        self._env.set_task(self._task)

    def reset(self):
        return self._env.reset()

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space
