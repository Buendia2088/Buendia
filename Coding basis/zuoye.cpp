#include<iostream>
using namespace std;
#include<string>
#include<cstring>
#include<stdio.h>
#include<iomanip>
#include<cmath>
int main (){
	int a[5][3]={11,12,11,995,779,925,885,565,990,895,875,605},e[3]={0,1,2};
	string b[2][3]={"Zhang","Yang","Liang","10001","10002","10003"};
	int i,j,h,g=10;
	double c[4];
	string d[2];
	cin>>d[1]>>c[0]>>d[0]>>c[1]>>c[2]>>c[3];
	for (i=0;i<3;i++){
		if (d[1]==b[1][i]){
			b[0][i]=d[0];
			for (j=0;j<4;j++){
				if(j!=0)
				a[j][i]=c[j]*10;
				else
					a[j][i]=c[j];
			}
			g=i;
		}
	}
	for (i=0;i<3;i++){
                a[4][i]=a[1][i]+a[2][i]+a[3][i];
        }
	for (i=0;i<2;i++){
		for(j=0;j<2-i;j++){
			if(a[0][e[i]]>a[0][e[i+1]]){
				h=e[i];
				e[i]=e[i+1];
				e[i+1]=h;
			}
		}
	}
	for(i=0;i<2;i++){
		for(j=0;j<2-i;j++){
			if(a[0][e[i]]==a[0][e[i+1]]&&a[4][e[i]]<a[4][e[i+1]]){
				h=e[i];
                                e[i]=e[i+1];
                                e[i+1]=h;
			}
		}
	}
	for (i=0;i<3;i++){
		if(i>0&&a[0][e[i]]==a[0][e[i-1]]){
			cout<<endl<<"  ";
		}
		else{
			cout<<endl;
			cout<<a[0][e[i]];
		}
		cout<<" "<<b[1][e[i]]<<" "<<b[0][e[i]];
		for (j=1;j<4;j++){
			cout<<" ";
			c[j]=a[j][e[i]];
			cout.setf(ios::fixed);
			cout<<setprecision(1);
			cout<<c[j]/10;
		}
		if (e[i]==g)
			cout<<" "<<"modified";
	}
}







	
