import gym 
import tempfile 
import numpy as np

from gym import wrappers

tdir = tempfile.mkdtemp()
env = gym.make('LunarLander-v2')
env = wrappers.Monitor(env, tdir, force=True)

def pid_function(env, observation):
	target_angle = observation[0]*0.7 + observation[2]*1.0

	if target_angle >  0.5: target_angle = 0.5
	if target_angle < -0.5: target_angle = -0.5
	target_y = np.abs(observation[0])

	angle_PID = (target_angle - observation[4]) - (observation[5])

	y_PID = (target_y - observation[1])*0.5 - (observation[3])*0.5

	if observation[6] or observation[7]:
		angle_PID = 0
		y_PID = -(observation[3]) * 0.5

	action = 0 
	if y_PID > np.abs(angle_PID) and y_PID > 0.05: action = 2
	elif angle_PID < -0.05: action = 3
	elif angle_PID > +0.05: action = 1

	return action


#100 trails for landing
for t in range(100):
	observation = env.reset()

	while 1 :
		env.render()
		action = pid_function(env, observation)

		observation, reward, done, info = env.step(action)

		if done:
			print("Episode finished after {} timesteps".format(t))
			break
env.close()

