
import gymnasium
import functools

from pokemon_pinball_gym import PokemonPinballEnv

import pufferlib.emulation
import pufferlib.postprocess


def env_creator(name='pokemon_pinball'):
    return functools.partial(make, name)

def make(name, headless: bool = True, state_path=None, buf=None):
    '''Pokemon Pinball'''
    env = PokemonPinballEnv(config={'headless': headless,'grayscale':False})
    #env = RenderWrapper(env)
    
    #env = gymnasium.wrappers.FrameStack(env, 4)
    
    env = pufferlib.postprocess.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)

class RenderWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        self.env = env

    @property
    def render_mode(self):
        return 'rgb_array'

    def render(self):
        return self.env.screen.screen_ndarray()