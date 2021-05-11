#include <iostream>
using namespace std;

class __shared_count{
    __shared_count(const __shared_count&);
    __shared_count& operator=(const __shared_count&);

    protected:
        long long  __shared_owners_; // how many owners do I have?
        virtual ~__shared_count();

    public:
        explicit __shared_count(long __refs = 0) noexcept : __shared_owners_(__refs){}

        void __add_shared() noexcept;

        bool __release_shared() noexcept;

};

class __shared_weak_count : private __shared_count {
    long __shared_weak_owners_;

public:
    explicit __shared_weak_count(long __refs = 0) noexcept 
        : __shared_count(__refs), 
          __shared_weak_owners_(__refs) {}

protected:
    virtual ~__shared_weak_count();

public:
    void __add_shared() noexcept;
    void __add_weak() noexcept;
    void __release_shared() noexcept;
    void __release_weak() noexcept;
    long use_count() const noexcept { return __shared_count::use_count();}

private:
    virtual void __on_zero_shared_weak() noexcept = 0;
};