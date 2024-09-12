/* depcheck.cpp - Dependency checker for NMake Makefiles
 * Copyright (c) 2024 Mika T. Lindqvist
 */

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: depcheck Makefile <top_directory>\n");
    return -1;
  }
  std::filebuf fb;
  if (fb.open (argv[1],std::ios::in)) {
    std::istream is(&fb);
    std::string makefile = argv[1];
    std::string l, tmp, tmp2;
    while (is) {
      std::getline(is, l);
      while (l.back() == '\\') {
        std::getline(is, tmp);
        l.replace(l.length() - 1, 1, tmp);
      }
      size_t pos = l.find("obj:");
      if (pos != std::string::npos) {
         std::string objfile = l.substr(0, pos+3);
         printf("File: %s\n", objfile.c_str());
         std::vector<std::string> files;
         std::stringstream ss(l.substr(pos+4));
         while(getline(ss, tmp, ' ')){
           if (tmp != "" && tmp != "/") {
             files.push_back(tmp);
           }
         }
         for (auto it = files.begin(); it != files.end(); ++it) {
           printf("Dependency: %s\n", (*it).c_str());
         }
         if (!files.empty()) {
           std::filebuf fb2;
           std::string src = files[0];
           size_t pos2 = src.find("$(TOP)");
           if (pos2 != std::string::npos) {
             src.replace(pos2, 6, argv[2]);
           }
           printf("Source: %s\n", src.c_str());
           if (fb2.open(src.c_str(),std::ios::in)) {
             std::istream is2(&fb2);
             std::vector<std::string> includes;
             while (is2) {
               std::getline(is2, l);
               pos = l.find("#");
               if (pos != std::string::npos) {
                 pos2 = l.find("include");
                 size_t pos3 = l.find("\"");
                 if (pos2 != std::string::npos && pos3 != std::string::npos && pos2 > pos && pos3 > pos2) {
                   tmp = l.substr(pos3 + 1);
                   pos2 = tmp.find("\"");
                   if (pos2 != std::string::npos) {
                     tmp = tmp.substr(0, pos2);
                   }
                   pos2 = tmp.find("../");
                   if (pos2 != std::string::npos) {
                     tmp = tmp.substr(3);
                   }
                   printf("Line: %s\n", tmp.c_str());
                   int found = 0;
                   for (size_t i = 1; i < files.size(); i++) {
                     pos3 = files[i].find("$(SUFFIX)");
                     if (pos3 != std::string::npos) {
                       tmp2 = files[i].substr(0, pos3).append(files[i].substr(pos3 + 9));
                       printf("Comparing dependency \"%s\" and \"%s\"\n", tmp2.c_str(), tmp.c_str());
                       if (tmp2 == tmp) {
                         printf("Dependency %s OK\n", tmp.c_str());
                         found = 1;
                         includes.push_back(tmp);
                         break;
                       }
                       printf("Comparing dependency \"%s\" and \"$(TOP)/%s\"\n", tmp2.c_str(), tmp.c_str());
                       if (tmp2 == std::string("$(TOP)/").append(tmp)) {
                         printf("Dependency %s OK\n", tmp.c_str());
                         found = 1;
                         includes.push_back(tmp);
                         break;
                       }

                       tmp2 = files[i].substr(0, pos3).append("-ng").append(files[i].substr(pos3 + 9));
                       printf("Comparing dependency \"%s\" and \"%s\"\n", tmp2.c_str(), tmp.c_str());
                       if (tmp2 == tmp) {
                         printf("Dependency %s OK\n", tmp.c_str());
                         found = 1;
                         includes.push_back(tmp);
                         break;
                       }
                       printf("Comparing dependency \"%s\" and \"$(TOP)/%s\"\n", tmp2.c_str(), tmp.c_str());
                       if (tmp2 == std::string("$(TOP)/").append(tmp)) {
                         printf("Dependency %s OK\n", tmp.c_str());
                         found = 1;
                         includes.push_back(tmp);
                         break;
                       }
                     } else {
                       printf("Comparing dependency \"%s\" and \"%s\"\n", files[i].c_str(), tmp.c_str());
                       if (files[i] == tmp) {
                         printf("Dependency %s OK\n", tmp.c_str());
                         found = 1;
                         includes.push_back(tmp);
                         break;
                       }
                       printf("Comparing dependency \"%s\" and \"$(TOP)/%s\"\n", files[i].c_str(), tmp.c_str());
                       if (files[i] == std::string("$(TOP)/").append(tmp)) {
                         printf("Dependency %s OK\n", tmp.c_str());
                         found = 1;
                         includes.push_back(tmp);
                         break;
                       }
                       printf("Comparing dependency \"%s\" and \"$(TOP)/arch/%s\"\n", files[i].c_str(), tmp.c_str());
                       if (files[i] == std::string("$(TOP)/arch/").append(tmp)) {
                         printf("Dependency %s OK\n", tmp.c_str());
                         found = 1;
                         includes.push_back(tmp);
                         break;
                       }
                       printf("Comparing dependency \"%s\" and \"$(TOP)/arch/generic/%s\"\n", files[i].c_str(), tmp.c_str());
                       if (files[i] == std::string("$(TOP)/arch/generic/").append(tmp)) {
                         printf("Dependency %s OK\n", tmp.c_str());
                         found = 1;
                         includes.push_back(tmp);
                         break;
                       }
                       printf("Comparing dependency \"%s\" and \"$(TOP)/arch/arm/%s\"\n", files[i].c_str(), tmp.c_str());
                       if (files[i] == std::string("$(TOP)/arch/arm/").append(tmp)) {
                         printf("Dependency %s OK\n", tmp.c_str());
                         found = 1;
                         includes.push_back(tmp);
                         break;
                       }
                       printf("Comparing dependency \"%s\" and \"$(TOP)/arch/x86/%s\"\n", files[i].c_str(), tmp.c_str());
                       if (files[i] == std::string("$(TOP)/arch/x86/").append(tmp)) {
                         printf("Dependency %s OK\n", tmp.c_str());
                         found = 1;
                         includes.push_back(tmp);
                         break;
                       }
                       printf("Comparing dependency \"%s\" and \"$(TOP)/test/%s\"\n", files[i].c_str(), tmp.c_str());
                       if (files[i] == std::string("$(TOP)/test/").append(tmp)) {
                         printf("Dependency %s OK\n", tmp.c_str());
                         found = 1;
                         includes.push_back(tmp);
                         break;
                       }
                     }
                   }
                   // Skip irrelevant dependencies
                   if (tmp.substr(0, 9) == "arch/s390") found = 1;
                   if (tmp == "zlib-ng.h" && std::find(includes.begin(), includes.end(), "zlib.h") != includes.end()) found = 1;
                   if (found == 0) {
                     printf("%s: Dependency %s missing for %s!\n", makefile.c_str(), tmp.c_str(), objfile.c_str());
                     return -1;
                   }
                 }
               }
             }
             for (size_t i = 1; i < files.size(); i++) {
               int found = 0;
               tmp = files[i];
               printf("Dependency: %s\n", tmp.c_str());
               pos2 = tmp.find("$(TOP)");
               if (pos2 != std::string::npos) {
                 tmp = tmp.substr(7);
               }
               for (size_t j = 0; j < includes.size(); j++) {
                 pos2 = tmp.find("$(SUFFIX)");
                 if (pos2 != std::string::npos) {
                   std::string tmp1 = tmp.substr(0, pos2).append(tmp.substr(pos2 + 9));
                   printf("[%zd/%zd] Comparing dependency \"%s\" and \"%s\"\n", j, includes.size(), tmp1.c_str(), includes[j].c_str());
                   if (tmp1 == includes[j]) {
                     printf("Dependency %s OK\n", files[i].c_str());
                     found = 1;
                     break;
                   }
                   printf("[%zd/%zd] Comparing dependency \"%s\" and \"arch/%s\"\n", j, includes.size(), tmp1.c_str(), includes[j].c_str());
                   if (tmp1 == std::string("arch/").append(includes[j])) {
                     printf("Dependency %s OK\n", files[i].c_str());
                     found = 1;
                     break;
                   }
                   printf("[%zd/%zd] Comparing dependency \"%s\" and \"arch/generic/%s\"\n", j, includes.size(), tmp1.c_str(), includes[j].c_str());
                   if (tmp1 == std::string("arch/generic/").append(includes[j])) {
                     printf("Dependency %s OK\n", files[i].c_str());
                     found = 1;
                     break;
                   }
                   printf("[%zd/%zd] Comparing dependency \"%s\" and \"arch/arm/%s\"\n", j, includes.size(), tmp1.c_str(), includes[j].c_str());
                   if (tmp1 == std::string("arch/arm/").append(includes[j])) {
                     printf("Dependency %s OK\n", files[i].c_str());
                     found = 1;
                     break;
                   }
                   printf("[%zd/%zd] Comparing dependency \"%s\" and \"arch/x86/%s\"\n", j, includes.size(), tmp1.c_str(), includes[j].c_str());
                   if (tmp1 == std::string("arch/x86/").append(includes[j])) {
                     printf("Dependency %s OK\n", files[i].c_str());
                     found = 1;
                     break;
                   }
                   printf("[%zd/%zd] Comparing dependency \"%s\" and \"test/%s\"\n", j, includes.size(), tmp1.c_str(), includes[j].c_str());
                   if (tmp1 == std::string("test/").append(includes[j])) {
                     printf("Dependency %s OK\n", files[i].c_str());
                     found = 1;
                     break;
                   }
                   tmp1 = tmp.substr(0, pos2).append("-ng").append(tmp.substr(pos2 + 9));
                   printf("[%zd/%zd] Comparing dependency \"%s\" and \"%s\"\n", j, includes.size(), tmp1.c_str(), includes[j].c_str());
                   if (tmp1 == includes[j]) {
                     printf("Dependency %s OK\n", files[i].c_str());
                     found = 1;
                     break;
                   }
                   printf("[%zd/%zd] Comparing dependency \"%s\" and \"arch/%s\"\n", j, includes.size(), tmp1.c_str(), includes[j].c_str());
                   if (tmp1 == std::string("arch/").append(includes[j])) {
                     printf("Dependency %s OK\n", files[i].c_str());
                     found = 1;
                     break;
                   }
                   printf("[%zd/%zd] Comparing dependency \"%s\" and \"arch/generic/%s\"\n", j, includes.size(), tmp1.c_str(), includes[j].c_str());
                   if (tmp1 == std::string("arch/generic/").append(includes[j])) {
                     printf("Dependency %s OK\n", files[i].c_str());
                     found = 1;
                     break;
                   }
                   printf("[%zd/%zd] Comparing dependency \"%s\" and \"arch/arm/%s\"\n", j, includes.size(), tmp1.c_str(), includes[j].c_str());
                   if (tmp1 == std::string("arch/arm/").append(includes[j])) {
                     printf("Dependency %s OK\n", files[i].c_str());
                     found = 1;
                     break;
                   }
                   printf("[%zd/%zd] Comparing dependency \"%s\" and \"arch/x86/%s\"\n", j, includes.size(), tmp1.c_str(), includes[j].c_str());
                   if (tmp1 == std::string("arch/x86/").append(includes[j])) {
                     printf("Dependency %s OK\n", files[i].c_str());
                     found = 1;
                     break;
                   }
                   printf("[%zd/%zd] Comparing dependency \"%s\" and \"test/%s\"\n", j, includes.size(), tmp1.c_str(), includes[j].c_str());
                   if (tmp1 == std::string("test/").append(includes[j])) {
                     printf("Dependency %s OK\n", files[i].c_str());
                     found = 1;
                     break;
                   }
                 } else {
                   printf("[%zd/%zd] Comparing dependency \"%s\" and \"%s\"\n", j, includes.size(), tmp.c_str(), includes[j].c_str());
                   if (tmp == includes[j]) {
                     printf("Dependency %s OK\n", files[i].c_str());
                     found = 1;
                     break;
                   }
                   printf("[%zd/%zd] Comparing dependency \"%s\" and \"arch/%s\"\n", j, includes.size(), tmp.c_str(), includes[j].c_str());
                   if (tmp == std::string("arch/").append(includes[j])) {
                     printf("Dependency %s OK\n", files[i].c_str());
                     found = 1;
                     break;
                   }
                   printf("[%zd/%zd] Comparing dependency \"%s\" and \"arch/generic/%s\"\n", j, includes.size(), tmp.c_str(), includes[j].c_str());
                   if (tmp == std::string("arch/generic/").append(includes[j])) {
                     printf("Dependency %s OK\n", files[i].c_str());
                     found = 1;
                     break;
                   }
                   printf("[%zd/%zd] Comparing dependency \"%s\" and \"arch/arm/%s\"\n", j, includes.size(), tmp.c_str(), includes[j].c_str());
                   if (tmp == std::string("arch/arm/").append(includes[j])) {
                     printf("Dependency %s OK\n", files[i].c_str());
                     found = 1;
                     break;
                   }
                   printf("[%zd/%zd] Comparing dependency \"%s\" and \"arch/x86/%s\"\n", j, includes.size(), tmp.c_str(), includes[j].c_str());
                   if (tmp == std::string("arch/x86/").append(includes[j])) {
                     printf("Dependency %s OK\n", files[i].c_str());
                     found = 1;
                     break;
                   }
                   printf("[%zd/%zd] Comparing dependency \"%s\" and \"test/%s\"\n", j, includes.size(), tmp.c_str(), includes[j].c_str());
                   if (tmp == std::string("test/").append(includes[j])) {
                     printf("Dependency %s OK\n", files[i].c_str());
                     found = 1;
                     break;
                   }
                 }
               }
               // Skip indirect dependencies
               if (tmp.find("arm_features.h") != std::string::npos
                   && std::find(includes.begin(), includes.end(), "cpu_features.h") != includes.end()
                   && (makefile.find(".arm") != std::string::npos
                      || makefile.find(".a64") != std::string::npos)) found = 1;
               if (tmp.find("x86_features.h") != std::string::npos
                   && std::find(includes.begin(), includes.end(), "cpu_features.h") != includes.end()
                   && makefile.find(".msc") != std::string::npos) found = 1;
               //
               if (tmp.find("generic_functions.h") != std::string::npos
                   && std::find(includes.begin(), includes.end(), "arch_functions.h") != includes.end()) found = 1;
               if (tmp.find("arm_functions.h") != std::string::npos
                   && std::find(includes.begin(), includes.end(), "arch_functions.h") != includes.end()
                   && (makefile.find(".arm") != std::string::npos
                      || makefile.find(".a64") != std::string::npos)) found = 1;
               if (tmp.find("x86_functions.h") != std::string::npos
                   && std::find(includes.begin(), includes.end(), "arch_functions.h") != includes.end()
                   && makefile.find(".msc") != std::string::npos) found = 1;
               if (found == 0) {
                 printf("%s: Dependency %s not needed for %s\n", makefile.c_str(), files[i].c_str(), objfile.c_str());
                 return -1;
               }
             }
             fb2.close();
           }
         }
      }
    }
    fb.close();
  }
  return 0;
}
