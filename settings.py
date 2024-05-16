class Settings:
    def __init__(self):
        # alpha, gamma
        self.learning_rate = 0.001  # learning_rate
        self.gamma = 0.99  # discount rate

        # Epsilon
        self.epsilon_max = 1.0  # maximum random rate (initial)
        self.epsilon_min = 0.01  # minimum random rate
        self.epsilon_min_episode_ratio = 0.75  # When epsilon should reach epsilon_min, ratio of trainings

        # Switch to allow Pacman to take "stop" action or not
        self.allow_stopping = False

        # Randomize food during training
        self.food_random = 0.4  # Chance that a food remains
        self.scale_food_reward = True  # Scale the food reward by 1/food_ranom

        # Random start position
        self.random_start_position = False
        self.random_start_has_food = True  # Allow starting also where there is food (food is removed)

        # All maps are padded to this size
        self.state_target_dimensions = (32, 32)

        # Buffer, sync and batch
        self.replay_buffer_size = 10000
        self.sync_target_episode_count = 100
        self.batch_size = 32

        # Network architecture
        # Convolution layers. Each will be followed by: batchnorm, relu, maxpool(2)
        self.conv_layers = [64, 64, 64, 64]
        self.dense_layers = [512, 64]
        self.dense_dropout = 0.5

        # Game and reward settings
        self.SCARED_TIME = 40  # Moves ghosts are scared, default = 40
        self.TIME_PENALTY = 1  # Number of points lost each round, default = 1
        self.FOOD_REWARD = 10  # Default = 10
        self.LOSE_PENALTY = 100  # default = 500
        self.WIN_REWARD = 500  # default = 500
        self.EAT_GHOST_REWARD = 100  # default = 200

        # If agent lose many points (from highest score), it dies
        self.DROP_FROM_HIGHWATER_DEATH = 50  # How big points drop to die
        self.DROP_FROM_HIGHWATER_DEATH_PENALTY = 50  # Penalty for dying in this way

        # If number of food is less, the rewards will look worse. This will mitigate that to get comparable numbers.
        if self.scale_food_reward:
            self.FOOD_REWARD = int(self.FOOD_REWARD / self.food_random)
