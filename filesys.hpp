//
//  filesys.hpp
//  CppCNNHandwriting
//
//  Created by Fabian Schneider on 23.04.19.
//  Copyright © 2019 Fabian Schneider. All rights reserved.
//

#ifndef filesys_hpp
#define filesys_hpp

#include <iostream>
#include <vector>

#include <dirent.h>
#include <sys/types.h>

namespace filesys {
    std::vector<std::string> ScanDirectoryByFiletype (std::string path, std::string format);
}

#endif /* filesys_hpp */
