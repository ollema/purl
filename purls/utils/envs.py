import multiprocessing as mp

import gym

from .logs import debug


class SubprocVecEnv(gym.Env):
    """
    Based on OpenAI's baselines/common/vec_env/subproc_vec_env.py
    """

    def __init__(self, envs):
        self.waiting = False
        self.closed = False

        nenvs = len(envs)
        ctx = mp.get_context("spawn")
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(nenvs)])
        self.ps = [
            ctx.Process(target=worker, args=(work_remote, remote, env))
            for (work_remote, remote, env) in zip(self.work_remotes, self.remotes, envs)
        ]
        debug("starting workers")
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(("get_spaces_spec", None))
        self.observation_space, self.action_space, self.spec = self.remotes[0].recv()

    def step(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True
        results = zip(*[remote.recv() for remote in self.remotes])
        self.waiting = False
        return results

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(("reset", None))
        return [remote.recv() for remote in self.remotes]

    def render(self):
        self._assert_not_closed()
        if len(self.remotes) == 1:
            self.remotes[0].send(("render", None))
            return self.remotes[0].recv()
        for remote in self.remotes:
            remote.send(("render", "rgb"))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        if self.closed:
            return
        self.close_extras()
        self.closed = True

    def close_extras(self):
        self.closed = True
        if self.waiting:
            debug("waiting for worker(s) to finish")
            for remote in self.remotes:
                try:
                    remote.recv()
                except (EOFError, ConnectionResetError):
                    continue
        debug("stopping workers")
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except BrokenPipeError:
                continue
        for p in self.ps:
            p.join()

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"

    def __del__(self):
        if not self.closed:
            self.close()


def worker(remote, parent_remote, env):
    parent_remote.close()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, done, info = env.step(data)
                if done:
                    obs = env.reset()
                remote.send((obs, reward, done, info))
            elif cmd == "reset":
                obs = env.reset()
                remote.send(obs)
            elif cmd == "render":
                if data == "rgb":
                    remote.send(env.render(mode="rgb_array"))
                else:
                    env.render()
                    remote.send(None)
            elif cmd == "close":
                remote.close()
                break
            elif cmd == "get_spaces_spec":
                remote.send((env.observation_space, env.action_space, env.spec))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
