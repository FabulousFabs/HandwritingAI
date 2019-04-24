//
//  filesys.cpp
//  CppCNNHandwriting
//
//  Created by Fabian Schneider on 23.04.19.
//  Copyright © 2019 Fabian Schneider. All rights reserved.
//

#include "filesys.hpp"

using namespace filesys;

std::vector<std::string> filesys::ScanDirectoryByFiletype (std::string path, std::string format) {
    std::vector<std::string> files;
    
    DIR* dirp = opendir(path.c_str());
    struct dirent *dp;
    
    while ((dp = readdir(dirp)) != NULL) {
        const std::string fn = std::string(dp->d_name);
        
        if (fn.substr(fn.find_last_of(".") + 1) == format) {
            files.push_back(fn);
        }
    }
    
    closedir(dirp);
    
    return files;
}
