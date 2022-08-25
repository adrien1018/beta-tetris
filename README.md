# BetaTetris

An actor-critic neural network agent playing NES Tetris with tapping speed and reaction time limitations.

This is the left well & improved version of the agent featured in [Classic Tetris AI Exhibition](https://www.twitch.tv/videos/1073802901).

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
python3 fceux.py model/model.pth --hz-avg=12 --start-level=18 --microadj-delay=21 --step-points=360 [--drought-mode]
```
and wait until it outputs `Ready`.

Load the Lua script `lua/tetris.lua` in FCEUX, and it will start playing 18-start games. To make it play other formats such as drought mode or 19/29 starts, re-run the Python server with different arguments, and modify the Lua script accordingly.

Available parameters are listed below:
| Item | Options | Description |
| ----- | ----------- | ------- |
| `hz-avg` | 12,13.5,15,20,30 | Hz |
| `start-level` | 18,19,29 | |
| `microadj-delay` | 8,16,21,25,61 | frames |
| `drought-mode` | (specify or not) | Game genie code: `TAOPOPYA` `APOPXPEY` |
| `step-points` | 100,360,2000 | Adjust aggression |

## Parameters & Statistics

The model is trained mainly for 12 & 20 Hz / 350ms reaction time. The optimal parameters & statistics of this agent for the 4 formats in AI Exhibition is as follows:

| Format |  Drought mode  |  Killscreen  |  Human possible  |  No limits  |  12 Hz Killscreen  |
| --- |  ---  |  ---  |  ---  |  ---  |  ---  |
| Optimal `step-points` | 360 | 100 | 360 | 2000 | 360 |
| Average         | 651,144 | 1,157,282 | 1,135,835 | 2,027,194 |  48,372 |
| Std. deviation  |  85,832 |   264,685 |   137,824 |   211,327 |  37,756 |
| 1st percentile  | 330,480 |    90,000 |   481,080 |   837,800 |   3,600 |
| 5th percentile  | 518,040 |   462,000 |   977,440 | 1,819,640 |   9,000 |
| 10th percentile | 552,620 |   948,740 | 1,023,740 | 1,890,300 |  12,600 |
| 30th percentile | 618,620 | 1,147,900 | 1,107,900 | 1,997,460 |  24,000 |
| 50th percentile | 658,200 | 1,218,820 | 1,150,440 | 2,051,940 |  38,400 |
| 70th percentile | 694,180 | 1,283,120 | 1,192,620 | 2,111,020 |  57,000 |
| 90th percentile | 743,980 | 1,368,700 | 1,258,400 | 2,192,480 |  96,600 |
| 95th percentile | 773,280 | 1,398,020 | 1,293,500 | 2,220,560 | 125,400 |
| 99th percentile | 816,940 | 1,458,440 | 1,352,620 | 2,271,640 | 186,600 |

Note that this statistics is collected using the same RNG used in training. The average score will be slightly higher on the real NES RNG. For example, the average of Human Possible format on NES RNG is 1,149,681.
