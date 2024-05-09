from learningAgents import ReinforcementAgent
import random


class randomAgent(ReinforcementAgent):
    def getAction(self, state):
        """
        state: can call state.getLegalActions()
        Choose an action and return it.
        """

        action = random.choice(state.getLegalActions())
        self.doAction(state, action)
        return action

    def update(self, state, action, nextState, reward):
        pass

    # Added recorded to function
    def registerInitialState(self, state, recorded):
        # This will be called at the beginning of each episode
        self.startEpisode()
        if self.episodesSoFar == 0:
            print("Beginning %d episodes of Training" % (self.numTraining))
