#include <iostream>
#include <string>
int main()
{
    using namespace std;
    string words;
    getline(cin, words);
    string done = "done";
    int i = 0,j = 0, prepos;
    int secondbreak = 0;

    while(cin)
    {
        prepos = 0;
        while(i < words.length())
        {
            if(words[i] == ' ')
            {
                if(words.substr(prepos, i - prepos) == done)
                {
                    secondbreak = 1;
                    break;
                }
                else
                {
                    prepos = i + 1;
                }
                j++;
            }
            i++;
        }
        if(words.substr(prepos, i - prepos) == done || secondbreak)
        {
            break;
        }
            else
            {
                j++;
            }
        getline(cin, words);
        i = 0;
        prepos = 0;
    }
    cout << "You entered a total of " << j << " words.";
    return 0;
}
        
    
