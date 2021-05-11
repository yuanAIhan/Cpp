#include <string.h>
#include <cstring>


class String{
    public:
        String(const char *value = "");
        String& operator=(const String& rhs)
        {
            if(this == &rhs) return *this;
            delete[] data;
            data = new char[strlen(rhs.data) + 1];
            strcpy(data, rhs.data);
            return *this;
        }

    private:
        char *data;
};