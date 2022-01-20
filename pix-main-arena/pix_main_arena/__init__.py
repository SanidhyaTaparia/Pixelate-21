from gym.envs.registration import register

register(
    id='pix_main_arena-v0',
    entry_point='pix_main_arena.envs:PixArena',
)
