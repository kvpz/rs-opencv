#include <iostream>
#include <fcntl.h>
#include <sys/stat.h>
#include <mqueue.h>
#include <cstring>
#include <unistd.h>  // for sleep function


int main() {
  //const char* mq_name = "/test_queue";
    const char* mq_name = "/object_detection_queue";
    const int mq_max_size = 10000;
    const int mq_msg_size = 102400;
    unsigned int priority;
    struct timespec timeout;

    // Create the message queue
    mqd_t mqd = mq_open(mq_name, O_CREAT | O_RDWR, 0666, nullptr);
    if (mqd == -1) {
        std::cerr << "Error creating message queue: " << strerror(errno) << std::endl;
        return 1;
    }

    timeout.tv_sec = 0;
    timeout.tv_nsec = 0;
    
    while(1) {
      char buffer[mq_msg_size];
      memset(buffer, 0, mq_msg_size);
      if (mq_receive(mqd, buffer, mq_max_size, nullptr) == -1) { //mq_receive(mqd, buffer, mq_max_size, nullptr)
        std::cerr << "Error receiving message from queue: " << strerror(errno) << std::endl;
        mq_close(mqd);
        return 1;
      }
      std::cout << "Received message: " << buffer << std::endl;
      
    }
    mq_close(mqd);
    mq_unlink(mq_name);
    return 0;
}

