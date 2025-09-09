#include "resolve.h"
#include <iostream>

#define FOUND_ALL_DEF 0
#define MULTI_DEF 1
#define NO_DEF 2

std::string errSymName;

int callResolveSymbols(std::vector<ObjectFile> &allObjects);

void resolveSymbols(std::vector<ObjectFile> &allObjects){
    int ret = callResolveSymbols(allObjects);
    if(ret == MULTI_DEF) {
        std::cerr << "multiple definition for symbol " << errSymName << std::endl;
        abort();
    } else if(ret == NO_DEF) {
        std::cerr << "undefined reference for symbol " << errSymName << std::endl;
        abort();
    }
}

int callResolveSymbols(std::vector<ObjectFile> &allObjects) {
    for(auto &obj : allObjects) {
        for(auto &re : obj.relocTable) {
            Symbol *currentSym = re.sym;
            std::string symName = currentSym->name;
            int strongCount = 0;
            int weakCount = 0;
            Symbol *firstStrong = nullptr;
            Symbol *firstWeak = nullptr;
            for(auto &checkedObj : allObjects) {
                for(auto &checkedSym : checkedObj.symbolTable) {
                    if(checkedSym.name != symName) continue;
                    if(checkedSym.bind == STB_GLOBAL &&
                        checkedSym.index != SHN_UNDEF && checkedSym.index != SHN_COMMON) {
                        strongCount++;
                        if(!firstStrong) firstStrong = &checkedSym;
                    }
                    else if(checkedSym.bind == STB_GLOBAL && checkedSym.index == SHN_COMMON) {
                        weakCount++;
                        if(!firstWeak) firstWeak = &checkedSym;
                    }
                }
            }
            if(strongCount > 1) {
                errSymName = symName;
                return MULTI_DEF;
            }
            else if(strongCount == 1) {
                re.sym = firstStrong;
            }
            else if(weakCount >= 1) {
                re.sym = firstWeak;
            }
            else {
                errSymName = symName;
                return NO_DEF;
            }
        }
    }
    return FOUND_ALL_DEF;
}