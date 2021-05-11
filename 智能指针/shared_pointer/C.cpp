#include <iostream>
#include "B.cpp"

class View
{
    public:
        int c;
        B *record;
        View(): record(nullptr){}
        View(int num)
        {
            c = num;
            record = new B;
            record->increment();
        }

        View(const View& obj)
        {
            record = obj.record;
            int count = record -> get_use();
            if(count != 0) 
            {
                record -> increment();
            }
            
            std::cout << "const View& obj" << std::endl;
        }

        View& operator=(const View& obj)
        {
            std::cout << "operator=" << std::endl;
            return *this;
        }
        ~View()
        {
            int number = record ->get_use();
            if(number == 0)
            {
                return;
            }
            record -> decrement();
            number = record -> get_use();
            if(number > 0) 
            {
                return;
            }
            if (record != nullptr)
            {
                delete record;
            }
            delete record;
        }
};

int main()
{
    View A(10);
    std::cout << A.record->get_use() << std::endl;
    View B = A;
    std::cout << A.record->get_use() << std::endl;
    std::cout <<" " << B.record->get_use() << std::endl;
    return 0;

}