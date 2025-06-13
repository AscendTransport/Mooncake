#include <iostream>
#include <string>
#include <fstream>
#include <json/json.h>
int main()
{
    std::ifstream file("/etc/hccl_16p.json");
    if (!file) {
        std::cerr << "无法打开文件hccl_16p.json" << std::endl;
        return -1;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json_str = buffer.str();
    Json::Value root;
    Json::Reader reader;
    reader.parse(json_str, root);


    std::string version = root["version"].asString();
    std::cout << "version:" << version;
    std::string server_count = root["server_count"].asString();
    std::cout << "server_count:" << server_count;
    // std::string server_id = root["server_list"]["server_id"].asString();
    // std::cout << "server_id:" << server_id;
    return 0;
}