#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <boost/asio.hpp>
#include <json/json.h>
#include <mqueue.h>
#include <vector>
#include <algorithm>
#include <sstream>

boost::asio::io_service io;
boost::asio::serial_port port(io);
std::string last_command_sended;

// Create or open a message queue
const char* queue_name = "/object_detection_queue";
struct mq_attr attr = {
    .mq_flags = 0,
    .mq_maxmsg = 10,
    .mq_msgsize = 1024,
};

struct Object {
    std::string object;
    int x1;
    int x2;
    int distance;
};

void send_command(const std::string& command){
    if (last_command_sended != command){
         port.write_some(boost::asio::buffer(command));
    } else {
        std::cout << "command already ran: " << command << std::endl;
    }
    if (command != "a" && command != "b" && command != "c" && command != "d" && command != "e" && command != "f"){
        last_command_sended = command;
    }
}

bool find_objects(const std::vector<Object>& objects, bool* object_located, bool* task_completed) {
    std::vector<Object> ducks;
    for (const auto& object_viewed : objects) {
        if (object_viewed.object == "duck") {
            ducks.push_back(object_viewed);
        }
    }

    if (!ducks.empty()) {
        *object_located = true;
        const Object& duck = *std::min_element(ducks.begin(), ducks.end(),
            [](const Object& x, const Object& y) { return x.distance < y.distance; });
        const int x1 = duck.x1;
        const int x2 = duck.x2;
        const int distance = duck.distance;
        const int center = x1 + (x2 - x1) / 2;
        std::cout << "distance: " << distance << std::endl;
        if (center < 300) {
            send_command("Z");
        } else if (center > 550) {
            send_command("C");
        } else if (center < 380) {
            send_command("R");
        } else if (center > 468) {
            send_command("L");
        } else if (distance > 20) {
            send_command("F");
            if (distance > 40) {
                send_command("c");
            } else {
                send_command("a");
            }
        } else {
            std::cout << "founded" << std::endl;
            send_command("S");
            *task_completed = true;
        }
    } else {
        send_command("S");
    }
}

int main(){
    last_command_sended = "";
    // Open the ACM0 port with the desired settings
    const std::string serial_port = "/dev/ttyACM0";
    const int baud_rate = 115200; // Adjust the baud rate according to your device's requirements
    const int timeout = 1; // Adjust the timeout as needed
    // boost::asio::io_service io;
    // boost::asio::serial_port port(io);
    port.open(serial_port);
    port.set_option(boost::asio::serial_port_base::baud_rate(baud_rate));
    port.set_option(boost::asio::serial_port_base::character_size(8));
    port.set_option(boost::asio::serial_port_base::parity(boost::asio::serial_port_base::parity::none));
    port.set_option(boost::asio::serial_port_base::stop_bits(boost::asio::serial_port_base::stop_bits::one));
    port.set_option(boost::asio::serial_port_base::flow_control(boost::asio::serial_port_base::flow_control::none));
    std::this_thread::sleep_for(std::chrono::seconds(2)); // Give the device some time to initialize if needed
    
    mqd_t mq = mq_open(queue_name, O_RDONLY | O_CREAT, 0644, &attr);
    if (mq == -1) {
        std::cerr << "Failed to create/open message queue: " << queue_name << std::endl;
        exit(1);
    }

    bool object_located = false;
    bool task_completed = false;
    int16_t ducks_founded_counter = 0; //to avoid false positive i am expecting to get 3 times a duck object
    int16_t ducks_not_founded_counter = 0;
    int16_t counter_to_find_target = 0;

    send_command("c");
    std::this_thread::sleep_for(std::chrono::seconds(1));

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        char message[40000];
        unsigned int priority;
        ssize_t bytes_received = mq_receive(mq, message, 40000, &priority);
        std::cout << "message: " << bytes_received << std::endl;
        if (bytes_received == -1) {
            std::cerr << "mq_receive() failed: " << std::strerror(errno) << std::endl;
        }
        if (bytes_received > 0) {
            message[bytes_received] = '\0';
            std::string decoded_message = std::string(message);

            //verify if we get duck input at least 3 times
            bool duck_in_image = message.find("duck") != std::string::npos;
            if (duck_in_image) {
                ducks_founded_counter++;
                if (ducks_founded_counter > 2){
                    ducks_not_founded_counter = 0;
                }
                if (ducks_founded_counter > 3000){ //to avoid overflow
                    ducks_founded_counter = 4;
                }
            } else {
                ducks_not_founded_counter++;
                if (ducks_not_founded_counter > 2){
                    ducks_founded_counter = 0;
                }
                if (ducks_not_founded_counter > 3000){ //to avoid overflow
                    ducks_not_founded_counter = 4;
                }
            }
            
            if (ducks_founded_counter > 2){
                counter_to_find_target = 0;
                // Convert the string to a list of dictionaries
                Json::Value root;
                Json::CharReaderBuilder builder;
                const std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
                std::string errors;
                bool success = reader->parse(decoded_message.c_str(), decoded_message.c_str() + decoded_message.size(), &root, &errors);
                if (!success) {
                    std::cerr << "Failed to parse JSON message: " << errors << std::endl;
                    continue;
                }
                std::vector<Object> objects;
                for (Json::Value& object : root) {
                    Object obj;
                    obj.object = object["object"].asString();
                    obj.x1 = object["x1"].asInt();
                    obj.x2 = object["x2"].asInt();
                    obj.distance = object["distance"].asInt();
                    objects.push_back(obj);
                }

                if (!task_completed) {
                    task_completed = find_objects(objects, &object_located, &task_completed);
                } else {
                    for (const auto& object_viewed : objects) {
                        if (object_viewed.object == "duck" && object_viewed.distance < 20) {
                           break; // next action (call arm function "arm.get(distance);")
                        } else {
                            task_completed = false;
                        }
                    }
                }
            }
            if (ducks_not_founded_counter > 2){
                if (counter_to_find_target % 7 == 3) {
                    if (object_located){
                        send_command('B');
                    } else (object_located) {
                        send_command('Z');
                    }
                } else if (counter_to_find_target % 7 == 6) {
                    send_command('S');                
                }
                counter_to_find_target++;
                // to avoid overflow
                if (counter_to_find_target > 5600){
                    counter_to_find_target = 1;
                }
            }
        }
    }
}
