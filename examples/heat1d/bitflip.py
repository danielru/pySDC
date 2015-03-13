#
# Simulates a bitflip in the first n digits of floor(x).
#
import math
import random

def bitflip(x, n):
  # Remove digits after decimal point
  x_int  = math.trunc(x)
  # Save the rest
  x_rest = x - float(x_int)
  # Convert integer part of x to binary
  x_bin  = bin(x_int)[2:]
  # Find a random position
  posi = random.randint(0,n)
  # Flip bit in random position
  if x_bin[posi]=='0':
    x_flip = '0b' + x_bin[0:posi] + '1' + x_bin[posi+1:]
  else:
    x_flip = '0b' + x_bin[0:posi] + '0' + x_bin[posi+1:]
  # Convert binart with fliped bit back to double and add the rest
  return int(x_flip, 2)+x_rest