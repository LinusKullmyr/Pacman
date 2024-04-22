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
import util

# import math
import torch
import collections
import time
import pytorch_lightning as pl
import logging


class QLearningAgent(ReinforcementAgent):
    """
    Q-Learning Agent

    Functions you should fill in:
      - computeValueFromQValues
      - computeActionFromQValues
      - getQValue
      - getAction
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions for a state
    """

    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)

        self.alpha = 0.0001  # learning_rate
        self.gamma = 1.0  # discount rate
        self.epsilon_min = 0.1  # minimum random rate
        self.epsilon_max = 0.5
        self.epsilon_scale = self.numTraining * 0.75
        self.replay_buffer_size = 10000
        self.batch_size = 32

        self.sync_target_training_count = 1000
        self.num_training_loops = 0
        self.sync_next_gameover = False

        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)  # Set the logging level
        self.trainer = pl.Trainer(max_epochs=1, enable_progress_bar=False)

        self.double_Q = None  # defer the creation of Q-nets until we have a state, in registerInitialState
        self.replay_buffer = collections.deque(maxlen=self.replay_buffer_size)
        self.action_tuples = (
            game.Directions.STOP,
            game.Directions.NORTH,
            game.Directions.EAST,
            game.Directions.SOUTH,
            game.Directions.WEST,
        )

        self.direction_to_index_map = {"Stop": 0, "North": 1, "East": 2, "South": 3, "West": 4}
        self.index_to_direction_map = {0: "Stop", 1: "North", 2: "East", 3: "South", 4: "West"}

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
        # if self.episodesSoFar == 0:
        #     print("Beginning %d episodes of Training" % (self.numTraining))

        # Create Q-nets (once)
        if self.double_Q is None:
            print("Creating Q-nets")
            num_ghosts = len(state.getGhostStates())
            input_channels = 4 + num_ghosts * 5
            output_size = 5
            self.double_Q = qnets.DQN(input_channels, output_size, self.gamma, self.alpha)

    # def getQValue(self, state, action):
    #     """
    #     Returns Q(state,action)
    #     Should return 0.0 if we have never seen a state
    #     or the Q node value otherwise
    #     """
    #     "*** YOUR CODE HERE ***"

    #     print("getQvalue")
    #     # util.raiseNotDefined()

    # def computeValueFromQValues(self, state):
    #     """
    #     Returns max_action Q(state,action)
    #     where the max is over legal actions.  Note that if
    #     there are no legal actions, which is the case at the
    #     terminal state, you should return a value of 0.0.
    #     """
    #     "*** YOUR CODE HERE ***"

    #     print("computeValueFromQValues")
    #     # util.raiseNotDefined()

    # def computeActionFromQValues(self, state):
    #     """
    #     Compute the best action to take in a state.  Note that if there
    #     are no legal actions, which is the case at the terminal state,
    #     you should return None.
    #     """
    #     "*** YOUR CODE HERE ***"

    #     print("computeActionFromQValues")
    #     # util.raiseNotDefined()

    def stateToTensor(self, state):
        """
        Take a gamestate and makes a tensor

        Parameters
        ----------
        state : GameState objects (from pacman.py)

        Returns
        -------
        state_tensor : torch tensor of dim [8+num_ghosts*5, H, W]

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
            # Offset by 4 to account for walls, food, capsules, and Pac-Man channels
            ghost_channel = 4 + i * 4 + direction_map[gdir]

            # print("ghosts", i, ghost_channel, int(gpos[1]), int(gpos[0]))
            x, y = gpos
            state_tensor[ghost_channel, int(x), int(y)] = 1

        # print(state_tensor)
        return state_tensor.unsqueeze(0)  # Unsqeeze to get a batch of 1

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

        legal_action_ints = [self.direction_to_index_map[x] for x in legalActions]

        ## Set current epsilon
        if self.epsilon == 0:  # Epsilon is set to 0 after numTraining epsiodes
            current_epsilon = 0
        elif self.numTraining > 0:  # If we are doing training
            current_epsilon = max(self.epsilon_min, self.epsilon_max * (1 - self.episodesSoFar / self.epsilon_scale))
        else:
            current_epsilon = 0

        ## Determine action
        # Note! Currently only allows legal actions!
        if random.random() < current_epsilon:
            # Random exploration with probability epsilon
            action_idx = random.choice(legal_action_ints)
        else:
            # Get action from Q-network
            state_tensor = self.stateToTensor(state)
            self.double_Q.eval()
            with torch.no_grad():
                q_values = self.double_Q(state_tensor).squeeze(0)

            assert q_values.shape == (5,)
            illegal_actions_mask = torch.ones_like(q_values).bool()
            illegal_actions_mask[legal_action_ints] = False
            q_values[illegal_actions_mask] = float("-inf")

            action_idx = torch.argmax(q_values)

        return self.action_tuples[action_idx]

    def reinforce(self, batch):
        states, actions, next_states, rewards, dones = zip(*batch)

        # Create tensors for the batch
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

        # Create dataset and DataLoader
        dataset = torch.utils.data.TensorDataset(states, actions, rewards, next_states, dones)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Run training
        self.trainer.fit(self.double_Q, loader)
        self.num_training_loops += 1
        if self.num_training_loops % self.sync_target_training_count == 0:
            self.sync_next_gameover = True

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

        # Make tensor of everything
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

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

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
            print("Reinforcement Learning Status:")
            windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
            if self.episodesSoFar <= self.numTraining:
                trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
                print("\tCompleted %d out of %d training episodes" % (self.episodesSoFar, self.numTraining))
                print("\tAverage Rewards over all training: %.2f" % (trainAvg))
            else:
                testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
                print("\tCompleted %d test episodes" % (self.episodesSoFar - self.numTraining))
                print("\tAverage Rewards over testing: %.2f" % testAvg)
            print("\tAverage Rewards for last %d episodes: %.2f" % (NUM_EPS_UPDATE, windowAvg))
            print("\tEpisode took %.2f seconds" % (time.time() - self.episodeStartTime))
            self.lastWindowAccumRewards = 0.0
            self.episodeStartTime = time.time()

        if self.episodesSoFar == self.numTraining:
            msg = "Training Done (turning off epsilon and alpha)"
            print("%s\n%s" % (msg, "-" * len(msg)))

        ## above is from parent class!

        if self.sync_next_gameover:
            print(f"Episode {self.episodesSoFar}: Synchronizing policy and target networks")
            self.syncNetworks()
            self.sync_next_gameover = False


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.001, gamma=0.9, alpha=0.1, numTraining=10, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args["epsilon"] = epsilon
        args["gamma"] = gamma
        args["alpha"] = alpha
        args["numTraining"] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
    ApproximateQLearningAgent

    You should only have to overwrite getQValue
    and update.  All other QLearningAgent functions
    should work as is.
    """

    def __init__(self, extractor="IdentityExtractor", **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
        Should return Q(state,action) = w * featureVector
        where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
        Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
