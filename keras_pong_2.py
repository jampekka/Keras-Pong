import gym
import numpy as np
import itertools

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Reshape, Dropout, Input
from keras.optimizers import Adam, Adamax, RMSprop
import keras.backend as K

import cv2
import sys

def pong_preprocess_screen(I):
    # Get only the game area as
    # black and white pixels
    I = I[35:195] 
    I = I[::2,::2,0] 
    I[I == 144] = 0 
    I[I == 109] = 0 
    I[I != 0] = 1
    return I.astype(np.float).ravel()

def get_model(input_shape, output_shape, learning_rate=0.001):
    # You can load a saved model by giving it as a parameter
    if len(sys.argv) > 1:
        model = load_model(sys.argv[1])
        return model, model
    inp = Input(shape=input_shape)
    
    # The zero-hidden-layer seems to learn quite fast when
    # presented with hand-fed features. Add layers and CPU time
    # if you make the input more complex
    model = Sequential([
        #Dense(10, activation="linear", input_shape=input_shape),
        #Dropout(0.2),
        #Dense(100, activation="elu"),
        #Dropout(0.2),
        Dense(np.prod(output_shape), activation='softmax', input_shape=input_shape)
        ])
    
    #model.add(Dense(200, activation='elu', input_shape=input_shape))
    #model.add(Dense(200, activation='elu'))
    #model.add(Dense(np.prod(output_shape), activation='softmax'))
    #model.add(Reshape(output_shape))
    
    #def policy_loss(y_true, y_pred):
    #    loglik = K.log(y_true * (y_true - y_pred) + (1 - y_true) * (y_true + y_pred))
    #    return loglik #*reward
    #    #return K.mean(loglik*rewards, axis=-1)

    opt = Adam(lr=learning_rate)
    #model = Model(inputs=[inp], outputs=out)
    #model.compile(loss=policy_loss, optimizer=opt)
    model.compile(loss="categorical_crossentropy", optimizer=opt)
    return model, model

possible_actions = np.array([
        0, # Nothing
        2, # Up
        5  # Down
        ])

class FeatEnv:
    def __init__(self):
        self.env = gym.make("Pong-v0")
        self.env.env.frameskip = 1
        self.env.frameskip = 1
        self.reset()
    
    def _get_feat(self):
        #idx = [4, 12, 21, 49, 50, 51, 54, 56, 58, 60, 64, 67, 121, 122]
        # This reads the paddle and ball positions directly from the
        # atari emulator's memory. Note that these are available even when
        # the ball is invisible between rounds (when it goes all the way up).
        # This hampers the learning somewhat as the paddle tends to go to up
        # in the beginning.
        idx = [54, 49, 21, 51]
        bally, ballx, oppy, playery = self.env.env._get_ram()[idx].astype(float)/206 - 0.5
        #playery += 38
        #oppy += 38
        #d = self.env.env._get_ram().copy()
        # Hand feed the difference between horizontal position between ball
        # and the player's paddle
        return np.array([
            #ballx, bally, oppy - bally,
            playery - bally
            ])


    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.prev_screen = self.current_screen
        #self.current_screen = pong_preprocess_screen(observation)
        self.current_screen = self._get_feat()
        #self.current_feat = np.hstack((self.current_screen, self.prev_screen))
        self.current_feat = self.current_screen
        return self.current_feat, reward, done, info
    
    def reset(self):
        #self.current_screen = pong_preprocess_screen(self.env.reset())
        self.env.reset()
        self.current_screen = self._get_feat()
        self.prev_screen = np.zeros(self.current_screen.shape)
        #self.current_feat = np.hstack((self.current_screen, self.prev_screen))
        self.current_feat = self.current_screen
        return self.current_feat

    def render(self):
        self.env.render()

import time
def main(render=True):
    learning_rate = 1e-3
    discount = 0.99
    env = gym.make("Pong-v0")
    
    env = FeatEnv()
    model, trainer = get_model(env.current_feat.shape, possible_actions.shape, learning_rate)
    cumulative_reward = 0.0
    n_actions = len(possible_actions)


    def play_round():
        rewards = []
        act_ps = []
        feats = []
        while True:
            if render:
                env.render()
                # If you want to see how the algorithm plays, adding a
                # sleep here helps
                #time.sleep(0.02)
            actprobs = model.predict(env.current_feat[np.newaxis], batch_size=1)[0]
            action = np.random.choice(n_actions, p=actprobs)
            
            # You can try a deterinistic control method like this
            """
            d = env.current_feat[0]
            if abs(d) < 0.03:
                action = 0
            elif d > 0:
                action = 1
            else:
                action = 2
            """

            feats.append(env.current_feat)
            feat, reward, done, info = env.step(possible_actions[action])
            rewards.append(reward)
            # One-hot encode the selected action.
            act_p = np.zeros(n_actions)
            act_p[action] = 1.0
            act_ps.append(act_p)
            # This would be something like advantage learning and was done
            # in the original, but I'm not sure this is the correct way of doing it,
            # so I just use the hot-one.
            #act_ps.append(act_p - actprobs[action])
            if reward != 0 or done:
                return rewards, act_ps, feats, done


    views = []
    actions = []
    
    batchlen = 1000
    update_freq = 20
    savelen = batchlen*1
    rounds_played = 0
    mean_reward = None

    batch_feats = []
    batch_targets = []
    batch_rewards = []
    batch_returns = []
    do_batch = lambda x: np.vstack(list(itertools.chain(*x)))
    while True:
        # Warning! Hacks here!
        rewards, acts, feats, done = play_round()
        rounds_played += 1

        step_rewards = np.zeros(len(rewards))
        running_reward = 0

        # The discounted reward is calculated here, but
        # it is not actually used
        for step in reversed(range(len(rewards))):
                running_reward *= discount
                running_reward += rewards[step]
                step_rewards[step] = running_reward

        # We actually use non-discounted per-round
        # mean reward for all of the round's actions
        round_reward = np.mean(rewards)
        step_rewards[:] = round_reward
        batch_rewards.append(rewards)
        #mean_reward += (round_reward - mean_reward)/rounds_played
        
        # This is just a running exponential average to
        # keep track of the progress. Not used in training
        if mean_reward is None:
            mean_reward = round_reward
        else:
            mean_reward = (1 - 0.99)*round_reward + 0.99*mean_reward
        
        #step_rewards -= np.mean(step_rewards)
        #step_rewards /= np.std(step_rewards)
        #round_target = np.array(step_rewards).reshape(-1, 1)*acts
        #round_target /= np.sum(round_target)
        #print(np.vstack(round_target))
        #print(mean_reward, round_reward)

        batch_feats.append(feats)
        batch_targets.append(acts)
        batch_returns.append(step_rewards)
        
        # Every update_freq train using the current
        # batch
        if rounds_played%update_freq == 0:
            batch_rewards = batch_rewards[-update_freq:]
            print(np.mean(do_batch(batch_rewards)))
            # Limit the batch to batchlen newest samples
            batch_feats = batch_feats[-batchlen:]
            batch_targets = batch_targets[-batchlen:]
            batch_returns = batch_returns[-batchlen:]

            r = do_batch(batch_returns).ravel()
            f = do_batch(batch_feats)
            a = do_batch(batch_targets)
            r -= np.mean(r)
            #r /= np.std(r)
            #print(r)
            #print(len(a))
            #a[r < 0,:] = 1 - a[r < 0,:]
            #a /= np.sum(a, axis=1).reshape(-1, 1)
            #trainer.train_on_batch(f, a, sample_weight=r)

            # sample_weight is used here instead of the weird multiplication
            # of thing in the original. I think this is (at least closer to) the
            # proper way of doing it in theoretical sense (as in policy gradient).
            trainer.fit(f, a, sample_weight=r)

            trainer.save(f"checkpoint.h5")

        if done:
            env.reset()
        #actprobs = model.predict(feats[np.newaxis], batch_size=1)[0]
        #action = np.random.choice(possible_actions, p=actprobs)
        #observation, reward, done, info = env.step(action)
        #feats = pong_preprocess_screen(observation)
        #print(info, reward)
        #input()
        #if done:
        #    env.reset()

if __name__ == '__main__':
    main()
