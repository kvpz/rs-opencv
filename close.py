import posix_ipc
import sys

# if len(sys.argv) != 2:
#     print("Usage: python close_queue.py /queue_name")
#     sys.exit(1)

queue_name = "/object_detection_queue"

try:
    mq = posix_ipc.MessageQueue(queue_name)
    mq.close()
    print(f"Message queue {queue_name} closed successfully.")
except posix_ipc.ExistentialError:
    print(f"Error: Message queue {queue_name} does not exist.")
    sys.exit(1)
except posix_ipc.PermissionsError:
    print(f"Error: Permission denied for message queue {queue_name}.")
    sys.exit(1)
print('Nothing')