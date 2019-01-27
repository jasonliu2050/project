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
}

int get() {
	int i=20;
	int j=100+i;
	
	return j;
}


int main() {
	
	for (int i = 0; i < 50; i++) {
		cout << "i="<<i<<endl;
		return 0;
	}
}
