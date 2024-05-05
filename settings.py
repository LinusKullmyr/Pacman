class Settings:
    def __init__(self):
        # alpha, gamma
        self.learning_rate = 0.001  # learning_rate
        self.gamma = 0.99  # discount rate

        # Epsilon
        self.epsilon_max = 1.0  # maximum random rate (initial)
        self.epsilon_min = 0.1  # minimum random rate
        self.epsilon_min_episode_ratio = 0.75  # When epsilon should reach epsilon_min, ratio of trainings

        # Switch to allow Pacman to take "stop" action or not
        self.allow_stopping = False

        # All maps are padded to this size
        self.state_target_dimensions = (32, 32)

        # Buffer, sync and batch
        self.replay_buffer_size = 1000
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
        self.LOSE_PENALTY = 100  # default = 500
        self.WIN_REWARD = 100  # default = 500
        self.EAT_GHOST_REWARD = 50  # default = 200

        # If agent lose this many points (from highest score), it dies
        self.DROP_FROM_HIGHWATER_DEATH = 50
