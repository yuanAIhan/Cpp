#include <iostream>
#include <memory>
#include <string>
using namespace std;

template <class T>
class my_shared_ptr
{
    private:
        T* m_ptr = nullptr;
        unsigned int* m_ref_count = nullptr;
    public:
        my_shared_ptr(): m_ptr(nullptr), m_ref_count(nullptr) {}
        my_shared_ptr(T* ptr): m_ptr(ptr), m_ref_count(new unsigned int(1)) {}
        
        my_shared_ptr(const my_shared_ptr& obj)
        {
            m_ptr = obj.m_ptr;
            m_ref_count = obj.m_ref_count;
            if(m_ref_count != nullptr)
            {
                (*m_ref_count)++;
            }   
        }

        my_shared_ptr& operator=(const my_shared_ptr & obj)
        {
            delete m_ptr;
            m_ptr = obj.m_ptr;
            delete m_ref_count;
            m_ref_count = obj.m_ref_count;
            
            if(m_ref_count!=nullptr)
            {
                (*m_ref_count)++;
            }
            return *this;
        }

        my_shared_ptr(my_shared_ptr && dying_obj): 
        m_ptr(nullptr),
        m_ref_count(nullptr)
        {
            //直接初始化后交换指针和引用计数, 等于清除了原shared_ptr的内容
            dying_obj.swap(*this);
        }

        my_shared_ptr & operator=(my_shared_ptr && dying_obj)
        {
            //my_shared_ptr(std::move(dying_obj))用移动构造函数创建出一个新的shared_ptr(此时dying_obj的内容被清除了)
            //再和this交换指针和引用计数
            //因为this的内容被交换到了当前的临时创建的my_shared_ptr里，原this指向的引用计数-1
            my_shared_ptr(std::move(dying_obj)).swap(*this);
            return *this;
        }

        void swap(my_shared_ptr & other)
        {
            std::swap(m_ptr, other.m_ptr);
            std::swap(m_ref_count, other.m_ref_count);
        }

        T* operator->() const
        {
            return m_ptr;
        }
        
        T& operator*() const
        {
            return m_ptr;
        }

        T* get() const
        {
            return m_ptr;
        }

        //以及获取引用计数
        unsigned int use_count() const
        {
            return *m_ref_count;
        }

         ~my_shared_ptr()
        {
            if(m_ref_count==nullptr)
            {
                return;
            }
            (*m_ref_count)--;
            if (*m_ref_count > 0)
            {
                return;
            }

            if (m_ptr != nullptr)
            {
                delete m_ptr;
            }
            delete m_ref_count;
        }
};


struct A{
    std::string m_str;
    A(std::string s): m_str(s)
    {
        cout <<"ctor:" << m_str<<endl;
    }
  
    ~A()
    {
        cout<<"dtor:"<<m_str<<endl;
    }
};


int main()
{
    cout<< "default constructor" <<endl;
    my_shared_ptr<A> empty_ptr;
    cout << (empty_ptr.use_count()==0) << endl;
    my_shared_ptr<A> a_ptr(new A("a"));
    cout << (a_ptr.use_count()==1) << endl;
    
    cout<< "copy constructor" <<endl;
    my_shared_ptr<A> copied_ptr(a_ptr);
    cout << (a_ptr.use_count()==2) << endl;
    cout << (copied_ptr.use_count()==2) << endl;
 
    cout<< "copy assignment" <<endl;
    my_shared_ptr<A> b_ptr(new A("b"));
    cout << (b_ptr.use_count()==1) << endl;
    b_ptr = a_ptr;
    cout << (a_ptr.use_count()==3) << endl;
    cout << (b_ptr.use_count()==3) << endl;
    
    cout<<"move constructor"<<endl;
    my_shared_ptr<A> move_ctor_ptr(std::move(a_ptr));
    cout << (move_ctor_ptr.use_count()==3) << endl;
    cout << (a_ptr.use_count()==0) << endl;
    
    cout<<"move assignment"<<endl;
    my_shared_ptr<A> b_ptr_observer = b_ptr;
    cout << (b_ptr.use_count()==4) << endl;
    cout << (b_ptr_observer.use_count()==4) << endl;
    
    my_shared_ptr<A> move_assign_ptr(new A("c"));
    my_shared_ptr<A> move_assign_ptr_observer=move_assign_ptr;
    cout << (move_assign_ptr.use_count()==2) << endl;
    cout << (move_assign_ptr_observer.use_count()==2) << endl;
    
    move_assign_ptr = std::move(b_ptr);
    cout << (b_ptr.use_count()==0) << endl;
    cout << (move_assign_ptr.use_count()==b_ptr_observer.use_count()) << endl;
    cout << (b_ptr_observer.use_count()==4) << endl;
    cout << (move_assign_ptr_observer.use_count()==1) << endl;
    
    return 0;
}