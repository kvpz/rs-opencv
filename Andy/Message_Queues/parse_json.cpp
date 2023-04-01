#include <iostream>
#include <string>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <mqueue.h>
#include <unistd.h>


#include <stack>
#include <vector>
#include <utility>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>


struct DetectedObject {
    std::string object;
    int x1;
    int y1;
    int x2;
    int y2;
    std::string inference_time;
    double distance;
};


int parse_data(std::string filename, struct DetectedObject &name){




  boost::property_tree::ptree pt;
  boost::property_tree::read_json(filename, pt);
  int x;
  int y;

  for (const auto& taskkey : pt) {        
      x = taskkey.second.get_child("x1").get_value<int>();
      y = taskkey.second.get_child("y1").get_value<int>();
  }

  std::cout<<"x1 "<<x<<std::endl;
  std::cout<<"x2 "<<y<<std::endl;


  return 0;

}

int main() {

    mqd_t mq = mq_open("/object_detection_queue", O_RDONLY | O_NONBLOCK);

    std::string json_data = "Hola";

    struct DetectedObject my_struct;

    

    


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
        parse_data(buffer,my_struct);
      }
      

      // std::this_thread::sleep_for(std::chrono::seconds(1));

    }

    

    return 0;
}