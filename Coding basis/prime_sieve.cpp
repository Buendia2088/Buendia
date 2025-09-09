#include<iostream>
#include<cmath>
#include<string.h>

using namespace std;
const int maxn=114514;
bool prime[maxn];

void judge_prime(int n)
{
	memset(prime,1,sizeof(prime));
	prime[0]=0;prime[1]=0;
	for(int i=2;i<n;i++)
	{
		if(prime[i]==1)
		for(int j=i*2;j<=n;j+=i)
		{
			prime[j]=0;
		}
	}	
}

int main()
{
	int n;
	cin>>n;
    if(n <= 2)
    {
        return 0;
    }
	judge_prime(n);
    cout << 2;
	for(int i=3;i<n;i++)
	{
		if(prime[i])
		{
            cout << " " << i;
        }
	} 
	return 0;
} 