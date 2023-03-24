#!/usr/bin/python3

import os
import posix_ipc
import time

the_path="/my_queue"

# try:
#   posix_ipc.unlink_message_queue(the_path)
# except Exception as e:
#   print("Error: "+str(e))


mq = posix_ipc.MessageQueue(the_path, flags=os.O_CREAT)

try:
    posix_ipc.SharedMemory(the_path)
    print("Queue created successfully")
except posix_ipc.ExistentialError:
    print("Queue already exists")

c=0
while True:
  c+=1
  msg='{}. Hello, C++!'.format(c)
  mq.send(msg.encode())

  print("Sent:",msg)

  time.sleep(0.5)

