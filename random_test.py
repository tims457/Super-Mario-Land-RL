from super_mario_land_rl.mario_env import SuperMarioLandEnv

env = SuperMarioLandEnv(render_mode="human")

for i in range(1):
    obs, _ = env.reset()
    for j in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done:
            break
    env.close()
