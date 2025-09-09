#include "relocation.h"

#include <sys/mman.h>
#include <iostream>
void handleRela(std::vector<ObjectFile> &allObject, ObjectFile &mergedObject, bool isPIE)
{
    /* When there is more than 1 objects, 
     * you need to adjust the offset of each RelocEntry
     */
    /* Your code here */
    if(allObject.size() > 1) {
        uint64_t textSize = 0;
        for(auto &obj : allObject) {
            for(auto &re : obj.relocTable) {
                re.offset += textSize;
            }
            textSize += obj.sections[".text"].size;
        }
    }
    /* in PIE executables, user code starts at 0xe9 by .text section */
    /* in non-PIE executables, user code starts at 0xe6 by .text section */
    uint64_t userCodeStart = isPIE ? 0xe9 : 0xe6;
    uint64_t textOff = mergedObject.sections[".text"].off + userCodeStart;
    uint64_t textAddr = mergedObject.sections[".text"].addr + userCodeStart;

    /* Your code here */
    for(auto &obj : allObject) {
        for(auto &re : obj.relocTable) {
            auto symValue = re.sym->value;
            uint64_t targetAddr = re.offset + textOff + (uint64_t)mergedObject.baseAddr;
            uint64_t newAddr = 0;
            if(re.type == 2 || re.type == 4) {
                newAddr = symValue - (re.offset + textAddr) + re.addend;
            }
            else {
                Section *sec = obj.sectionsByIdx[re.sym->index];
                newAddr = symValue + re.addend;
            }
            *reinterpret_cast<int *>(targetAddr) = newAddr;
        }
    }
}