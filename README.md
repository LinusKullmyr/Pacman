# Pacman

Double Q-learning implementation for PacMan

To run:

-l map
-p agent
-x training runs
-n test runs

-r to save models
--replay to use a saved model

-q for no graphics

Example:
python3 pacman.py -l smallGrid2.lay -p DQAgent -x 2500 -n 2510

Example for saving:
py pacman.py -l smallGrid2.lay -p DQAgent -x 5000 -n 5010 -r  

Example for loading:
py pacman.py -l smallGrid2.lay -p DQAgent --replay 'savedModels\saved-model-40015-6-14-37-22'

Replay best smallClassic:
python3 pacman.py -l smallClassic.lay -p DQAgent --replay 'used_logs/240516_041543_smallClassic/best_model.pt'

Replay best smallGrid:
python3 pacman.py -l smallGrid.lay -p DQAgent --replay 'used_logs/240514_144530_SmallGrid/best_model.pt'

Replay best meidumGrid:
python3 pacman.py -l mediumGrid.lay -p DQAgent --replay 'used_logs/240514_152055_MediumGrid/best_model.pt'


Change number of games played in replayGame function in pacman.py -> 753

Base code taken from https://berkeleyai.github.io/cs188-website/project3.html
