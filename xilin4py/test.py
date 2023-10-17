# Author: 赩林, xilin0x7f@163.com
import time

print("This is a long message.", end="", flush=True)
time.sleep(2)  # just to wait for a while
print("\rNow I'm at the beginning.", end="", flush=True)

