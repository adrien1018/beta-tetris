#define NO_PYTHON
#define DEBUG_METHODS
#include "game.h"

int main() {
  Tetris t;
  char buf[12];
  while (true) {
    printf("Input>");
    fflush(stdout);
    if (scanf("%s", buf) == -1) break;
    switch (buf[0]) {
      case 'p': t.PrintAllState(); break;
      case 'f': t.PrintState(true); break;
      case 's': t.PrintState(); break;
      case 'r': t.ResetRandom(1, 0.3, 0.0); break;
      case 'i': {
        int r, x, y;
        scanf("%d %d %d", &r, &x, &y);
        double rr = t.InputPlacement({r, x, y}).first;
        printf("Reward: %f\n", rr);
        break;
      }
    }
  }
  /*
  int field[][10] = {
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {},
    {0,0,1,0,0,0,0,0,0,0},
    {0,1,1,1,0,0,0,1,0,0},
    {1,1,1,1,1,0,1,1,1,1},
    {1,1,1,1,1,1,1,1,1,0},
    {1,1,1,1,1,1,1,1,1,0},
    {1,1,1,1,1,1,1,1,1,0},
    {1,1,1,1,1,1,1,1,0,1},
  };
  Tetris::Field r_field;
  for (int i = 0; i < 20; i++) for (int j = 0; j < 10; j++) r_field[i][j] = field[i][j];
  t.ResetGame(18, 12, 0, 21, 0, true, 0, 0, 1);
  t.SetState(r_field, 3, 5, {0, 12, 7}, 139, 401880, 362);
  t.InputPlacement({0, 13, 9});
  t.PrintAllState();
  t.InputPlacement({0, 12, 8});
  t.PrintAllState();
  t.InputPlacement({1, 12, 9});
  t.PrintAllState();
  t.InputPlacement({1, 14, 5});
  t.PrintAllState();
  */
}
