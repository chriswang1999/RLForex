"""
test and visualize trained cartpole agents
"""
import numpy as np
import torch
# from gym.envs.classic_control import rendering
# import time
# import skvideo.io
import functools

from environment import ForexEnv
from agents_old import RandomAgent
from agents_old import DQNAgent
from agents_old import Forex_reward_function
from feature import ForexIdentityFeature

def test(agent, environment, max_timesteps):
    """
    return observation and action data for one episode
    """
    # observation_history is a list of tuples (observation, termination signal)
    observation_history = [(environment.reset()[0],environment.reset()[1],environment.reset()[2], False)]
    action_history = []
    
    t = 0
    done = False
    while not done:
        action = agent.act(observation_history, action_history)
        timestamp, state, price_record, done = environment.step(action)
        action_history.append(action)
        observation_history.append((timestamp, state, price_record, done))
        t += 1
        done = done or (t == max_timesteps)

    return observation_history, action_history


# def renderCartpole(states, actions, mode='human'):
#     action_dict = {0: "left", 1: "nothing", 2: "right"}
#     x_threshold = 5
#
#     screen_width = 600
#     screen_height = 400
#
#     carty = screen_height / 2
#     cartwidth = screen_width/8#50.0
#     cartheight = 0.6 * cartwidth#30.0
#     polewidth = cartwidth/5
#     polelen = (3.2)*cartheight
#
#     wallwidth = polewidth
#     wallheight = screen_height
#
#     viewer = rendering.Viewer(screen_width, screen_height)
#     scale = (screen_width/2 - 2*cartwidth/2 - wallwidth)/x_threshold
#
#     l,r,t,b = -wallwidth/2, wallwidth/2, wallheight/2, -wallheight/2
#     wall = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
#     gray = 100/255
#     wall.set_color(gray, gray, gray)
#     walltrans = rendering.Transform()
#     walltrans.set_translation(x_threshold*scale +wallwidth/2 + cartwidth/2 + screen_width/2.0, carty)
#     wall.add_attr(walltrans)
#     viewer.add_geom(wall)
#     owall = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
#     owall.set_color(gray, gray, gray)
#     owalltrans = rendering.Transform()
#     owalltrans.set_translation(-x_threshold*scale -wallwidth/2 - cartwidth/2 + screen_width/2.0, carty)
#     owall.add_attr(owalltrans)
#     viewer.add_geom(owall)
#
#
#     l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
#     axleoffset =cartheight/4.0
#     cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
#     carttrans = rendering.Transform()
#     cart.add_attr(carttrans)
#     viewer.add_geom(cart)
#
#     l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
#     pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
#     pole.set_color(.8,.6,.4)
#     poletrans = rendering.Transform(translation=(0, axleoffset))
#     pole.add_attr(poletrans)
#     pole.add_attr(carttrans)
#     viewer.add_geom(pole)
#
#     axle = rendering.make_circle(polewidth/2)
#     axle.add_attr(poletrans)
#     axle.add_attr(carttrans)
#     axle.set_color(.5,.5,.8)
#     viewer.add_geom(axle)
#
#     right_arrow_points = [(0, 0), (-2, 1), (-2, 0.5), (-6, 0.5),
#                        (-6, -0.5), (-2, -0.5), (-2, -1), (0, 0)]
#     right_arrow_points = [(screen_width / 2 - cartwidth / 2 + cartwidth/8 * x,
#                            carty + cartheight / 2 + cartwidth/8 * y)
#                           for (x, y) in right_arrow_points]
#     right_arrow = rendering.FilledPolygon(right_arrow_points)
#     right_arrow.set_color(0, 0, 0)
#     right_arrow_trans = rendering.Transform()
#     right_arrow.add_attr(right_arrow_trans)
#
#
#     left_arrow_points = [(0, 0), (2, 1), (2, 0.5), (6, 0.5),
#                         (6, -0.5), (2, -0.5), (2, -1), (0, 0)]
#     left_arrow_points = [(screen_width / 2 + cartwidth / 2 + cartwidth/8 * x,
#                           carty + cartheight / 2 + cartwidth/8 * y)
#                          for (x, y) in left_arrow_points]
#     left_arrow = rendering.FilledPolygon(left_arrow_points)
#     left_arrow.set_color(0, 0, 0)
#     left_arrow_trans = rendering.Transform()
#     left_arrow.add_attr(left_arrow_trans)
#
#     track = rendering.Line((-x_threshold*scale - cartwidth/2 + screen_width/2.0,carty),
#                             (x_threshold*scale + cartwidth/2 + screen_width/2.0,carty))
#     track.set_color(0,0,0)
#     viewer.add_geom(track)
#     frames = []
#
#     for t in range(len(actions)):
#         x, x_dot, theta, theta_dot = states[t]
#         action = actions[t]
#         cartx = x*scale + screen_width/2.0 # MIDDLE OF CART
#         carttrans.set_translation(cartx, carty)
#         poletrans.set_rotation(-theta)
#         if action_dict[action] == "right":
#             right_arrow_trans.set_translation(x*scale, 0)
#             viewer.add_onetime(right_arrow)
#         if action_dict[action] == "left":
#             left_arrow_trans.set_translation(x*scale, 0)
#             viewer.add_onetime(left_arrow)
#
#         frames.append(viewer.render(return_rgb_array = mode=='rgb_array'))
#
#     viewer.close()
#     return np.array(frames).astype(np.uint8)
#

if __name__=='__main__':
    dqn_model_path = './dqn.pt'

    np.random.seed(321)
    torch.manual_seed(123)

    env = ForexEnv()

    agent = DQNAgent(
        action_set=[0, 1, 2],
        reward_function=functools.partial(Forex_reward_function),
        feature_extractor=ForexIdentityFeature(),
        test_model_path=dqn_model_path)

    observation_history, action_history = test(
        agent=agent, 
        environment=env, 
        max_timesteps=500)

    reward = np.sum(agent.get_episode_reward(observation_history, action_history))

    print('agent %s, cumulative reward %.2f' % (str(agent), reward))
    
    # tau = len(observation_history)
    # states = np.zeros((tau, 4))
    # for t in range(tau):
    #     states[t] = observation_history[t][0]
    #
    # frames = renderCartpole(states, action_history, mode='rgb_array')
    #
    # # save video
    # writer = skvideo.io.FFmpegWriter("./cartpole.mp4",
    #     outputdict={'-vcodec': 'libx264', '-b': '5000k', '-pix_fmt': 'yuv420p'},
    #     inputdict={'-framerate': str(1 / env.tau)})
    # for frame in frames:
    #     writer.writeFrame(frame)
    # writer.close()


