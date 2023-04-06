import serial
import time
import json

import posix_ipc

# Open the ACM0 port with the desired settings
serial_port = '/dev/ttyACM0'
baud_rate = 115200 # Adjust the baud rate according to your device's requirements
timeout = 1 # Adjust the timeout as needed
ser = serial.Serial(serial_port, baud_rate, timeout=timeout)
time.sleep(2) # Give the device some time to initialize if needed

# Create or open a message queue
queue_name = "/object_detection_queue"

try:
    mq = posix_ipc.MessageQueue(queue_name, flags=posix_ipc.O_RDONLY)
except posix_ipc.ExistentialError:
    print(f"Error: Message queue {queue_name} does not exist.")
    exit(1)


# This is a global variable to make sure we do not send twice the same command
last_command_sended = ''

def send_command(command:str):
    global last_command_sended
    # print(f'Command: {command}')
    if last_command_sended != command:
        ser.write(command.encode())
    else:
        print('command already ran :' + command )
        None
    if not command in ['a','b','c','d','e','f']:
        last_command_sended = command
    

def find_objects(objects):
    ducks = []
    for object_viewed in objects:
        if object_viewed['object'] == 'duck':
            ducks.append(object_viewed)

    if ducks:   
        duck = min(ducks, key=lambda x: x['distance'])
        x1, x2, distance = duck['x1'], duck['x2'], duck['distance']
        center = x1 + (x2-x1)//2
        print(f'distance: {distance}')
        if center < 300:
            send_command('Z')
        elif center > 550:
            send_command('C')
        elif center < 380:
            send_command('R')
        elif center > 468:
            send_command('L')
        elif distance > 20:
            send_command('F')
            if distance > 40:
                send_command('c')
            else:
                send_command('a')
            
        else:
            print ('founded')
            send_command('S')
            return True
    else:
        send_command('S')
    return False
    # return False

founded = False 

send_command('c')
time.sleep(1)
while True:
    time.sleep(0.01) 
    message, priority = mq.receive()
    decoded_message = message.decode()

    # Convert the string to a list of dictionaries
    objects = json.loads(decoded_message)
    if not founded:
        founded = find_objects(objects)
    else:
        break
    # if i:
    #     time.sleep(1)
    
print("DONE")
ser.close()
mq.close()