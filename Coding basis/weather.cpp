#include <bits/stdc++.h>
using namespace std;
#define int long long

struct Temperature {
    int day;
    int temperature;
};

bool compareByTemperature(Temperature a, Temperature b) {
    return a.temperature < b.temperature;
}

bool compareByDay(Temperature a, Temperature b) {
    return a.day < b.day;
}

int main() {
    int t;
    cin >> t;
    
    while (t--) {
        int n, k;
        cin >> n >> k;
        
        vector<Temperature> a(n+1);
        vector<int> b(n+1);
        
        for (int i = 1; i <= n; i++) {
            cin >> a[i].temperature;
            a[i].day = i;
        }
        
        for (int i = 1; i <= n; i++)
            cin >> b[i];
        
        sort(a.begin()+1, a.end(), compareByTemperature);
        sort(b.begin()+1, b.end());
        
        for (int i = 1; i <= n; i++)
            a[i].temperature = b[i];
        
        sort(a.begin()+1, a.end(), compareByDay);
        
        for (int i = 1; i <= n; i++)
            cout << a[i].temperature << " ";
        
        cout << "\n";
    }
    
    return 0;
}
