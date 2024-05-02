# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from learningAgents import ReinforcementAgent

# from featureExtractors import *
import game
import qnets
import random

# import util

# import math
import torch
import collections
import time

# import pytorch_lightning as pl
# import logging
import warnings


class DQAgent(ReinforcementAgent):
    """
    Q-Learning Agent
    """

    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)

        self.index = 0  # Agent index 0 is Pacman

        self.learning_rate = 0.001  # learning_rate
        self.gamma = 0.99  # discount rate
        self.epsilon_max = 1.0  # maximum random rate (initial)
        self.epsilon_min = 0.1  # minimum random rate
        self.epsilon_min_episode = int(self.numTraining * 0.75)  # When epsilon should reach epsilon_min
        self.replay_buffer_size = 200
        self.batch_size = 32

        self.sync_target_episode_count = 100

        self.state_target_dimesions = (32, 32)  # All maps are padded to this size
        self.double_Q = None  # defer the creation of Q-nets until we have a state, in registerInitialState
        self.replay_buffer = collections.deque(maxlen=self.replay_buffer_size)

        self.action_tuples = (
            game.Directions.NORTH,
            game.Directions.EAST,
            game.Directions.SOUTH,
            game.Directions.WEST,
            game.Directions.STOP,
        )

        self.direction_to_index_map = {
            "North": 0,
            "East": 1,
            "South": 2,
            "West": 3,
            "Stop": 4,
        }
        self.index_to_direction_map = {0: "North", 1: "East", 2: "South", 3: "West", 4: "Stop"}

        # Ignore specific UserWarnings from PyTorch about Lazy modules
        warnings.filterwarnings("ignore", message="Lazy modules are a new feature")

        # From parent class init
        # self.episodesSoFar = 0
        # self.accumTrainRewards = 0.0
        # self.accumTestRewards = 0.0
        # self.numTraining = int(numTraining)
        # self.epsilon = float(epsilon)
        # self.alpha = float(alpha)
        # self.discount = float(gamma)

    def syncNetworks(self):
        self.double_Q.update_target_network()

    def registerInitialState(self, state):
        # This will be called at the beginning of each episode
        self.startEpisode()
        if self.episodesSoFar == 0:
            print("Beginning %d episodes of Training" % (self.numTraining))

        # Create Q-nets (once)
        if self.double_Q is None:
            print("Creating Q-nets")
            num_ghosts = len(state.getGhostStates())
            input_channels = 4 + num_ghosts * 5
            output_size = 5
            self.double_Q = qnets.double_DQN(input_channels, output_size, self.learning_rate, self.gamma)

    def stateToTensor(self, state):
        """
        Take a gamestate and makes a tensor

        Parameters
        ----------
        state : GameState objects (from pacman.py)

        Returns
        -------
        state_tensor : torch tensor of dim [4 + num_ghosts * 5, W, H]

        """
        # Extracting the basic game state components
        walls = state.getWalls()  # class Grid, can be indexed [x][y]
        food = state.getFood()  # Same type as walls

        # Pac-Man state
        pacman_state = state.getPacmanState()
        ppos, _ = pacman_state.getPosition(), pacman_state.getDirection()

        # Ghost states
        ghosts = []
        for gs in state.getGhostStates():
            gpos = gs.getPosition()
            gdir = gs.getDirection()
            ghosts.append((gpos, gdir))
        num_ghosts = len(ghosts)

        # Capsules
        capsules = state.getCapsules()  # List of (x, y) tuples

        # Environment dimensions
        width, height = walls.width, walls.height

        # Initialize tensors for each channel
        num_channels = 4 + num_ghosts * 5  # walls, food, capsules, Pac-Man, 5 per ghost
        state_tensor = torch.zeros((num_channels, width, height), dtype=torch.float32)

        # Map for direction to tensor indices
        direction_map = self.direction_to_index_map

        # Fill the tensor
        for x in range(width):
            for y in range(height):
                state_tensor[0, x, y] = walls[x][y]  # Wall channel
                state_tensor[1, x, y] = food[x][y]  # Food channel

        # Capsule channel (as sparse locations)
        for x, y in capsules:
            state_tensor[2, x, y] = 1

        # Pac-Man channel
        x, y = ppos
        state_tensor[3, x, y] = 1

        # Ghost channels
        for i, (gpos, gdir) in enumerate(ghosts):
            # Offset by 4 to account for walls, food, capsules, Pac-Man channels
            ghost_channel = 4 + i * 4 + direction_map[gdir]

            # print("ghosts", i, ghost_channel, int(gpos[1]), int(gpos[0]))
            x, y = gpos
            state_tensor[ghost_channel, int(x), int(y)] = 1

        # Calculate padding
        # print(state_tensor.size())
        _, h, w = state_tensor.size()

        # Calculate padding to achieve target dimensions
        # Ensure the total padding size is evenly divisible by 2
        pad_height = max(32 - h, 0)
        pad_width = max(32 - w, 0)

        # Calculate padding for each side to maintain symmetry
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # Apply symmetric padding
        state_tensor = torch.nn.functional.pad(state_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)

        return state_tensor.unsqueeze(0)  # Unsqeeze to get a batch of 1

    def getCurrentEpsilon(self):
        ## TODO - this could probably be made nicer
        if self.epsilon == 0:  # Epsilon is set to 0 after numTraining epsiodes
            current_epsilon = 0
        elif self.numTraining > 0:  # If we are doing training
            current_epsilon = max(
                self.epsilon_min,
                self.epsilon_max - (self.epsilon_max - self.epsilon_min) / self.epsilon_min_episode * self.episodesSoFar,
            )
        else:
            current_epsilon = 0

        return current_epsilon

    def getAction(self, state):
        """
        Compute the action to take in the current state.  With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise.  Note that if there are
        no legal actions, which is the case at the terminal state, you
        should choose None as the action.

        The Agent will receive a GameState and
        must return an action from Directions.{North, South, East, West, Stop}
        """

        ## If no legal actions, return None
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None

        # Get indices of legal actions
        legal_action_ints = [self.direction_to_index_map[x] for x in legalActions]

        ## Set current epsilon
        current_epsilon = self.getCurrentEpsilon()

        ## Determine action
        # Note! Currently only allows legal actions!
        if random.random() < current_epsilon:
            # Random exploration with probability epsilon
            action_idx = random.choice(legal_action_ints)
        else:
            # Get action from Q-network
            state_tensor = self.stateToTensor(state)
            with torch.no_grad():
                q_values = self.double_Q.predict_eval(state_tensor).squeeze(0).detach()

            assert q_values.shape == (5,)

            illegal_actions_mask = torch.ones_like(q_values).bool()
            illegal_actions_mask[legal_action_ints] = False
            q_values[illegal_actions_mask] = float("-inf")

            action_idx = torch.argmax(q_values)

        action = self.action_tuples[action_idx]

        self.doAction(state, action)

        return action

    def reinforce(self, batch):
        states, actions, next_states, rewards, dones = zip(*batch)

        # Create stack/cat tensor for the batch
        states = torch.cat(states, dim=0)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.cat(next_states, dim=0)
        dones = torch.stack(dones)

        # print("state.shape", states.shape)
        # print("actions.shape", actions.shape)
        # print("rewards.shape", rewards.shape)
        # print("next_states.shape", next_states.shape)
        # print("dones.shape", dones.shape)

        batch = (states, actions, next_states, rewards, dones)

        self.double_Q.training_step(batch)

    def update(self, state, action, nextState, reward):
        """
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        You should do your Q-Value update here

        NOTE: You should never call this function,
        it will be called on your behalf
        """

        # This should tell us if the game is over or not
        if self.getLegalActions(nextState):
            done = False
        else:
            done = True

        # Make tensors of everything
        state_tensor = self.stateToTensor(state)
        action_tensor = torch.tensor([self.direction_to_index_map[action]], dtype=torch.long)
        nextState_tensor = self.stateToTensor(nextState)
        reward_tensor = torch.tensor([reward], dtype=torch.float32)
        done_tensor = torch.tensor([done], dtype=torch.bool)

        # Experience to replay buffer
        experience = (state_tensor, action_tensor, nextState_tensor, reward_tensor, done_tensor)
        self.replay_buffer.append(experience)

        # Run reinforcement if replay buffer is full
        if len(self.replay_buffer) >= self.replay_buffer_size:
            batch = random.sample(list(self.replay_buffer), self.batch_size)

            self.reinforce(batch)

    def final(self, state):
        """
        Called by Pacman game at the terminal state
        """
        deltaReward = state.getScore() - self.lastState.getScore()
        self.observeTransition(self.lastState, self.lastAction, state, deltaReward)
        self.stopEpisode()

        # Make sure we have this var
        if not "episodeStartTime" in self.__dict__:
            self.episodeStartTime = time.time()
        if not "lastWindowAccumRewards" in self.__dict__:
            self.lastWindowAccumRewards = 0.0
        self.lastWindowAccumRewards += state.getScore()

        NUM_EPS_UPDATE = 100
        if self.episodesSoFar % NUM_EPS_UPDATE == 0:
            # print(f"Episode: {self.episodesSoFar}")
            windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
            if self.episodesSoFar <= self.numTraining:
                trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
                print("Completed %d out of %d training episodes" % (self.episodesSoFar, self.numTraining))
                print("Average Rewards over all training: %.2f" % (trainAvg))
            else:
                testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
                print("Completed %d test episodes" % (self.episodesSoFar - self.numTraining))
                print("Average Rewards over testing: %.2f" % testAvg)
            print("Average Rewards for last %d episodes: %.2f" % (NUM_EPS_UPDATE, windowAvg))
            print("Episodes took %.2f seconds" % (time.time() - self.episodeStartTime))
            predict = "predict_eval"
            train = "training_step"
            calls_pred = self.double_Q.get_times(predict)[0]
            calls_train = self.double_Q.get_times(train)[0]
            time_pred = self.double_Q.get_times(predict)[1]
            time_train = self.double_Q.get_times(train)[1]
            self.double_Q.reset_times()
            print(f"Total number of predictions: {calls_pred}")
            print(f"Total time for {predict}: {time_pred:.2f} seconds")
            print(f"Total number of trainings: {calls_train}")
            print(f"Total time for {train}: {time_train:.2f} seconds")
            print(f"Current epsilon = {self.getCurrentEpsilon():.3f}")
            print("")
            self.lastWindowAccumRewards = 0.0
            self.episodeStartTime = time.time()

        if self.episodesSoFar == self.numTraining:
            msg = "Training Done (turning off epsilon and alpha)"
            print("%s\n%s" % (msg, "-" * len(msg)))

        if len(self.replay_buffer) == self.replay_buffer_size and self.episodesSoFar % self.sync_target_episode_count == 0:
            print(f"Episode {self.episodesSoFar}: Synchronizing policy and target networks\n")
            self.syncNetworks()
