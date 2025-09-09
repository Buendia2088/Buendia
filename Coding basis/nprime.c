#include<stdio.h>
#include<math.h>
int Primenumber(int num)
{
	int flag=0;
	int i=sqrt(num),j;
	j=2;
	while(j<=i)
	{
		if(num%j==0)
		{
			flag=1;
			break;
		}
		j++;
	}
	if(flag==1)
		return 0;//表示不是质数
	else
		return 1;
}
int main(){
	int N,count=0,flag,a[1000];
	int min=10,max,s; 
	int i;
	scanf("%d",&N);
	for(i=1;i<N-1;i++)
		min=min*10;
	max=min*10-1;
	for(;min<=max;min++)
	{	
		s=min;
		for(i=1;i<=N;i++)
		{
			flag=Primenumber(s);
			if(flag==0)
				break;
			s=s/10;
			if(s==1){
				flag=0;
				break;
			}
		}
		if(flag==1)
			a[count++]=min;
	}
	for(i=0;i<count;i++)
		printf("%d\n",a[i]);

		
    return 0;
}

