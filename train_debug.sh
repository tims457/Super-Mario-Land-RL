cd ~/dev/Super-Mario-Land-RL
source .venv/bin/activate
cd ~/dev/PufferLib
# python demo.py --mode train --env super_mario_land --vec serial --train.compile False --rnn-name Recurrent
python demo.py --mode train --env super_mario_land --vec multiprocessing --train.compile False