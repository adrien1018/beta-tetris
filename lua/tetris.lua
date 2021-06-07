gameStateAddress = 0x00C0
playStateAddress = 0x0048
selectLevelAddress = 0x0047
copyrightAddress1 = 0x00C3
nowTetriminoIDAddress = 0x0062
nextTetriminoIDAddress = 0x00BF

tetriminoXAddress = 0x0041
tetriminoYAddress = 0x0040
tetriminoRotateAddress = 0x0042

function enterFromMode()
  for i = 1,3 do
    emu.frameadvance()
  end
  for i = 1,3 do
    joypad.set(1, {down=true})
    emu.frameadvance()
    emu.frameadvance()
  end
  joypad.set(1, {start=true})
  emu.frameadvance()
end

function enterFromMain()
  for i = 1,4 do
    emu.frameadvance()
  end
  joypad.set(1, {start=true})
  emu.frameadvance()
  enterFromMode()
end

function enterFromCopyright()
  while memory.readbyteunsigned(gameStateAddress) == 0 do
    emu.frameadvance()
    if memory.readbyteunsigned(copyrightAddress1) == 0 then
      joypad.set(1, {start=true})
    end
  end
  enterFromMain()
end

function startGame(level)
  for i = 1,4 do
    emu.frameadvance()
  end
  local x = 9
  if level == 18 then
    x = 8
  end
  while true do
    local now = memory.readbyteunsigned(selectLevelAddress)
    if now == x then
      break
    elseif now < 5 then
      joypad.set(1, {down=true})
    elseif now < x then  
      joypad.set(1, {right=true})
    else
      joypad.set(1, {left=true})
    end
    emu.frameadvance()
    emu.frameadvance()
  end
  joypad.set(1, {A=true})
  emu.frameadvance()
  joypad.set(1, {A=true, start=true})
  emu.frameadvance()
  while memory.readbyteunsigned(playStateAddress) ~= 1 do
    emu.frameadvance()
  end
end

-- use O(n) queue anyway for simplicity
recvQueue = ""
sendQueue = ""

function trySend(tcp, msg)
  msg = msg or ""
  if string.len(msg) ~= 0 then
    sendQueue = sendQueue .. msg
  end
  if string.len(sendQueue) == 0 then
    return
  end
  local x, y, z = tcp:send(sendQueue)
  if x == nil then
    x = z
  end
  sendQueue = string.sub(sendQueue, x + 1)
end

function tryReceive(tcp, size)
  sizeToRead = math.max(0, size - string.len(recvQueue))
  if sizeToRead == 0 then
    local ret = string.sub(recvQueue, 1, size)
    recvQueue = string.sub(recvQueue, size + 1)
    return ret
  end
  local x, y, z = tcp:receive(sizeToRead)
  if x == nil then
    x = z
  end
  recvQueue = recvQueue .. x
  if string.len(sizeToRead) >= size then
    local ret = string.sub(recvQueue, 1, size)
    recvQueue = string.sub(recvQueue, size + 1)
    return ret
  else
    return nil
  end
end

function resetQueue(tcp)
  tcp:receive(1000000)
  sendQueue = ""
  recvQueue = ""
end

--[[
TCP stream format:
- Piece (1 byte): (0x00~0x06)
- Starting level (1 byte): (0x12 or 0x13)
- Piece position (3 bytes): [rotate](0x00~0x03) [x](0x00~0x13) [y](0x00~0x09)
- Move sequence (len+2 bytes): 0xfe [seq length] [seq...]
  - Each byte is a frame or'ed by following keys:
    - 0x01 (left)
    - 0x02 (right)
    - 0x04 (A)
    - 0x08 (B)
- Procedure
  - Game start (sent 4 bytes): 0xff [current piece] [next piece] [starting level]
  - Game loop
    - Current piece microadjustment sequence (appended to next piece move sequence) (receive)
    - Next piece move sequence (receive)
    - Locked position + next piece (sent 5 bytes): 0xfd [locked position] [next piece]
--]]

pieceMap = {}
pieceMap[2] = 0
pieceMap[7] = 1
pieceMap[8] = 2
pieceMap[10] = 3
pieceMap[11] = 4
pieceMap[14] = 5
pieceMap[18] = 6
--             T            J         Z    O     S         L         I
rotateMap = {3, 0, 1,  1, 2, 3, 0,  0, 1,  0,  0, 1,  3, 0, 1, 2,  1, 0}
rotateMap[0] = 2

function sendStartGame(tcp, level)
  local currentPiece = memory.readbyteunsigned(nowTetriminoIDAddress)
  local nextPiece = memory.readbyteunsigned(nextTetriminoIDAddress)
  local msg = string.char(0xff, pieceMap[currentPiece], pieceMap[nextPiece], level)
  print('startGame', currentPiece, nextPiece, level)
  trySend(tcp, msg)
end

function receiveSequence(tcp, seq)
  if not seq.length then
    local p = tryReceive(tcp, 2)
    if p then
      if string.byte(p, 1) == 0xfe then
        seq.length = string.byte(p, 2)
      end
    end
  end
  if seq.length and not seq[seq.length] then
    local p = tryReceive(tcp, seq.length)
    if p then
      for i = 1,seq.length do
        local x = string.byte(p, i)
        local buttons = {}
        if x % 2 >= 1 then buttons.left = true end
        if x % 4 >= 2 then buttons.right = true end
        if x % 8 >= 4 then buttons.A = true end
        if x % 16 >= 8 then buttons.B = true end
        seq[i] = buttons
      end
      return true
    end
  end
  return false
end

function receiveTwoSequence(tcp, curSeq, nextSeq)
  if nextSeq.length and nextSeq[nextSeq.length] then
    return
  end
  if curSeq.length and curSeq[curSeq.length] then
    receiveSequence(tcp, nextSeq)
  else
    receiveSequence(tcp, curSeq)
  end
end

function gameLoop(tcp, level)
  resetQueue(tcp)
  sendStartGame(tcp, level)
  local endGame = false
  local nextSequence = {length=1}
  nextSequence[1] = {}
  while not endGame do
    local curSequence = {}
    local fNextSequence = {}
    local inMicro = false
    local currentFrame = 1
    local st = memory.readbyteunsigned(playStateAddress)
    while st == 1 do
      trySend()
      receiveTwoSequence(tcp, curSequence, fNextSequence)
      if inMicro then
        if curSequence[currentFrame] then
          joypad.set(1, curSequence[currentFrame])
          currentFrame = currentFrame + 1
        end
      else
        if nextSequence[currentFrame] then
          joypad.set(1, nextSequence[currentFrame])
          currentFrame = currentFrame + 1
          if currentFrame > nextSequence.length then
            inMicro = true
            currentFrame = 1
          end
        end
      end
      emu.frameadvance()
      st = memory.readbyteunsigned(playStateAddress)
    end
    nextSequence = fNextSequence
    local rotate = rotateMap[memory.readbyteunsigned(tetriminoRotateAddress)]
    local x = memory.readbyteunsigned(tetriminoXAddress)
    local y = memory.readbyteunsigned(tetriminoYAddress)
    while st ~= 1 do
      if st == 10 then
        endGame = true
        break
      end
      emu.frameadvance()
      st = memory.readbyteunsigned(playStateAddress)
    end
    local nextPiece = memory.readbyteunsigned(nextTetriminoIDAddress)
    trySend(tcp, string.char(0xfd, rotate, x, y, pieceMap[nextPiece]))
  end
  while memory.readbyteunsigned(gameStateAddress) == 4 do
    joypad.set(1, {start=true})
    for i = 1,5 do
      emu.frameadvance()
    end
  end
end

local socket = require("socket")
local tcp = assert(socket.tcp())
local ret, msg = tcp:connect("localhost", 3456)
if not ret then
  print("Connection failed", msg)
  while true do emu.frameadvance() end
end
tcp:settimeout(0.007, 't')

emu.addgamegenie("XNEOOGEX")
emu.poweron()
enterFromCopyright()

while true do
  startGame(18)
  gameLoop(tcp, 18)
  startGame(19)
  gameLoop(tcp, 19)
end

