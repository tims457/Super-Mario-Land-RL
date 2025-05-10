cd ~/dev/Super-Mario-Land-RL
source .venv/bin/activate
cd ~/dev/PufferLib
# python demo.py --mode train --env super_mario_land --vec multiprocessing --track
python demo.py --mode train --env super_mario_land --vec multiprocessing --rnn-name Recurrent --track