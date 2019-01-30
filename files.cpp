#include <iostream>
#include <string>
#include <vector>
class base {
public:
	int age;
};
class d1 {
public:
	int d1;
};

void swap(int& a,int&b) {
	int tmp = a;
	a = b;
	b = tmp;
	cout<<endl;
}
void set() {
   int i=0;

}
int get() {
	int i=200;
	int j=100+i;
	
	return j;
}

// added from feature1
int main() {
// from linux add	
	for (int i = 0; i < 50; i++) {
		cout << "LINUX added<<endl;
		cout << "i="<<i<<endl;
		return 0;
	}
}
