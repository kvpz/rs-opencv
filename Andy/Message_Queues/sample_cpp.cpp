#include <iostream>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <mqueue.h>
#include <unistd.h>

#include <chrono>
#include <thread>
#include <csignal>

mqd_t mq = mq_open("/my_queue", O_RDONLY | O_NONBLOCK);



void signalHandler(int signal) {
    std::cout << "\nCTRL+C detected, terminating program..." << std::endl;
    mq_close(mq);
    exit(signal);
}

int main() {

    signal(SIGINT, signalHandler);

    if (mq == -1) {
        std::cerr << "Error opening message queue: " << strerror(errno) << std::endl;
        return 1;
    }

    // Receive a message from the queue
    struct mq_attr attr;
    mq_getattr(mq, &attr);
    size_t max_msg_size = attr.mq_msgsize;
    

    

    while (true){

      char buffer[max_msg_size];
      unsigned int prio;
      
      memset(buffer, 0, sizeof(buffer));

      ssize_t bytes_received = mq_receive(mq, buffer, sizeof(buffer), &prio);

      if (bytes_received == -1 && errno == EAGAIN){
        //std::cout<<"Queue not available. Waiting 1s..."<<std::endl;
        usleep(100000);;
        continue;
      }

      if (strlen(buffer)>0){
        std::cout << "Received message: " << buffer << std::endl;
      }
      

      // std::this_thread::sleep_for(std::chrono::seconds(1));

    }

    return 0;
}

