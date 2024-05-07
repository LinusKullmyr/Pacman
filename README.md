# Pacman

Double Q-learning implementation for PacMan

To run:

-l map
-p agent
-x training runs
-n test runs

-r to save models
--replay to use a saved model

Example:
python3 pacman.py -l smallGrid2.lay -p DQAgent -x 2500 -n 2510

Example for saving:
py pacman.py -l smallGrid2.lay -p DQAgent -x 5000 -n 5010 -r  

Example for loading:
py pacman.py -l smallGrid2.lay -p DQAgent --replay 'savedModels\saved-model-40015-6-14-37-22'


Base code taken from https://berkeleyai.github.io/cs188-website/project3.html
