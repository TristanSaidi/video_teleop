import argparse
import pickle

import numpy as np
from envs.ihm_env import IHMEnv
from model.model3d5fh import Model3D5FH
from sim.hand import Hand


def test(args):
    model_xml = Model3D5FH.toxml(object="cube")
    sim = Hand(
        model=model_xml,
        default_hand_joint_pos=np.array([0, 0.0, 0.0] * 5),
        default_object_pose=np.array([0, 0, 0.25, 1.0, 0.0, 0.0, 0.0]),
    )

    env = IHMEnv(
        sim=sim,
        max_episode_length=500,
        max_dq=0.05,
        discrete_action=False,
        randomize_initial_state=False,
    )

    action_space = env.action_space
    print("seeds", env.seed(1))

    # We will look at the random exploration behaviour after pickling and unpicking
    # because that may introduce additional bugs which we want to look for.
    env = pickle.loads(pickle.dumps(env))
    if args.render:
        env.render()
    while True:
        done = False
        env.reset()
        while not done:
            l1, l2, l3 = sim.get_link_lengths('finger1')
            action = np.array([[0, 0, 5*(l2+l3)]]*5)
            obs, _, done, info = env.step(action, cartesian=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", default=True, action="store_true")
    test(parser.parse_args())