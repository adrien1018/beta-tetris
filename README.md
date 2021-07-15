# BetaTetris

An actor-critic neural network agent playing NES Tetris with tapping speed and reaction time limitations.

This is the same agent featured in [Classic Tetris AI Exhibition](https://www.twitch.tv/videos/1073802901).

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
python3 fceux.py model/model.pth --hz-avg=12 --start-level=18 --microadj-delay=21 --game-over-penalty=-1 --first-gain=0.03 [--drought-mode]
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
| `game-over-penalty` | 0,-0.1,-1 | Adjust aggression |
| `first-gain` | -1 or >=0 | Additional value given to the actor output when doing 1-ply move search. Resonable value range is 0.02 to 0.1. -1 to disable move search. |

## Parameters & Statistics

The model is trained mainly on the 4 formats in AI Exhibition, so the performance in other settings will be slightly worse. Optimal parameters for the 4 formats:

| Format | `game-over-penalty` | `first-gain` |
| ----- | ----------- | ------- |
| Drought mode | -0.1 or -1 | 0.06 |
| Killscreen | -1 | -1 |
| Human possible | -1 | 0.03 |
| No limits | 0 | 0.03 |

The statistics of the 4 formats under the above parameters:

| Format |  Drought mode  |  Killscreen  |  Human possible  |  No limits  |
| --- |  ---  |  ---  |  ---  |  ---  |
| Average |  618,774  |  952,940  |  1,083,338  |  2,030,553  |
| Std. deviation |  132,927  |  449,421  |  148,174  |  220,232  |
| 1st percentile |  107,920  |  7,200  |  249,280  |  1,100,880  |
| 5th percentile |  369,060  |  108,000  |  801,500  |  1,644,640  |
| 10th percentile |  437,300  |  203,400  |  958,960  |  1,863,440  |
| 30th percentile |  592,300  |  718,800  |  1,058,840  |  1,994,200  |
| 50th percentile |  641,600  |  1,145,180  |  1,107,300  |  2,075,280  |
| 70th percentile |  693,180  |  1,279,280  |  1,153,200  |  2,135,520  |
| 90th percentile |  746,780  |  1,406,560  |  1,211,900  |  2,219,780  |
| 95th percentile |  778,460  |  1,455,740  |  1,239,360  |  2,248,220  |
| 99th percentile |  818,560  |  1,534,000  |  1,277,780  |  2,303,560  |
