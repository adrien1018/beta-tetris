# BetaTetris

An actor-critic neural network agent playing NES Tetris with tapping speed and reaction time limitations.

This is a version trained mainly on 20 Hz killscreen play, and it uses left well strategy.

## Setup

First, run the following command to build the library for core game logic (written in C++):
```
make -C tetris
```

Python requirements are listed in `requirements.txt`. Install them by
```
pip install -r requirements.txt
```

## Watch it play

Run the Python server first:
```
python3 fceux.py model/model.pth --hz-avg=20 --start-level=19 --microadj-delay=21 [--drought-mode]
```
and wait until it outputs `Ready`.

Load the Lua script `lua/tetris.lua` in FCEUX, and it will start playing 18-start games. To make it play other formats such as drought mode or 19/29 starts, re-run the Python server with different arguments, and modify the Lua script accordingly.

Available parameters are listed below:
| Item | Options | Description |
| ----- | ----------- | ------- |
| `hz-avg` | 20 | Hz |
| `start-level` | 29 | |
| `microadj-delay` | 21 | frames |
| `drought-mode` | (specify or not) | Game genie code: `TAOPOPYA` `APOPXPEY` |

## Statistics

Note that this statistics is collected using the same RNG used in training. The average score will be slightly higher on the real NES RNG.

| Format |  Killscreen  |
| --- |  ---  |
| Average |  1,170,992 |
| Std. deviation |  238,829 |
| 1st percentile |  95,400 |
| 5th percentile |  599,400 |
| 10th percentile | 983,480 |
| 30th percentile | 1,156,060 |
| 50th percentile | 1,220,360 |
| 70th percentile | 1,282,120 |
| 90th percentile | 1,371,260 |
| 95th percentile | 1,403,220 |
| 99th percentile | 1,464,920 |
