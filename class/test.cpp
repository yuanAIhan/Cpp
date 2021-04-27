#include <iostream>
using namespace std;


class name{
    public:
        int number = 10;
};
template <class MemorySpace = void, class DestroyFunctor = void>
class SharedAllocationRecord;

class SharedAllocationHeader {
 private:
  using Record = SharedAllocationRecord<void, void>;
  static constexpr unsigned maximum_label_length =
      (1u << 7 /* 128 */) - sizeof(Record*);  //120
  template <class, class>
  friend class SharedAllocationRecord;
  Record* m_record;
  char m_label[maximum_label_length];
 public:
  /* Given user memory get pointer to the header */
 static const SharedAllocationHeader* get_header(void* alloc_ptr) {
    return reinterpret_cast<SharedAllocationHeader*>(
        reinterpret_cast<char*>(alloc_ptr) - sizeof(SharedAllocationHeader));
  }
  const char* label() const { return m_label; }
};

template <>
class SharedAllocationRecord<void, void> {
 protected:
  static_assert(sizeof(SharedAllocationHeader) == (1u << 7 /* 128 */),
                "sizeof(SharedAllocationHeader) != 128");
  template <class, class>
  friend class SharedAllocationRecord; //共享分配记录 
  using function_type = void (*)(SharedAllocationRecord<void, void>*);

  SharedAllocationHeader* const m_alloc_ptr;
  size_t const m_alloc_size;
  function_type const m_dealloc;
  
#ifdef KOKKOS_ENABLE_DEBUG
  SharedAllocationRecord* const m_root;
  SharedAllocationRecord* m_prev;
  SharedAllocationRecord* m_next;
#endif
  int m_count;

  SharedAllocationRecord(SharedAllocationRecord&&)      = delete;
  SharedAllocationRecord(const SharedAllocationRecord&) = delete;
  SharedAllocationRecord& operator=(SharedAllocationRecord&&) = delete;
  SharedAllocationRecord& operator=(const SharedAllocationRecord&) = delete;

  /**\brief  Construct and insert into 'arg_root' tracking set.
   *         use_count is zero.
   */
  SharedAllocationRecord(
#ifdef KOKKOS_ENABLE_DEBUG
      SharedAllocationRecord* arg_root,
#endif
      SharedAllocationHeader* arg_alloc_ptr, size_t arg_alloc_size,
      function_type arg_dealloc);
      //这个应该是一个构造函数
 private:
  static  int t_tracking_enabled;     //返回的值有两种要么是这个要么是0的结果！
           
 public:
  virtual std::string get_label() const { return std::string("Unmanaged"); }

#ifdef KOKKOS_IMPL_ENABLE_OVERLOAD_HOST_DEVICE
  /* Device tracking_enabled -- always disabled */
  
  static int tracking_enabled() { return 0; }
#endif

  static int tracking_enabled() {
    KOKKOS_IMPL_IF_ON_HOST { return t_tracking_enabled; }
    else {
      return 0;
    }
  }

  /**\brief A host process thread claims and disables the
   *        shared allocation tracking flag.
   */
  static void tracking_disable() {
    KOKKOS_IMPL_IF_ON_HOST { t_tracking_enabled = 0; }
  }

  /**\brief A host process thread releases and enables the
   *        shared allocation tracking flag.
   */
  static void tracking_enable() {
    KOKKOS_IMPL_IF_ON_HOST { t_tracking_enabled = 1; }
  }

  virtual ~SharedAllocationRecord() = default;

  SharedAllocationRecord()
      : m_alloc_ptr(nullptr),
        m_alloc_size(0),
        m_dealloc(nullptr)
#ifdef KOKKOS_ENABLE_DEBUG
        ,
        m_root(this),
        m_prev(this),
        m_next(this)
#endif
        ,
        m_count(0) {}

  static constexpr unsigned maximum_label_length = SharedAllocationHeader::maximum_label_length;

  KOKKOS_INLINE_FUNCTION
  const SharedAllocationHeader* head() const { return m_alloc_ptr; }

  /* User's memory begins at the end of the header */
  KOKKOS_INLINE_FUNCTION
  void* data() const { return reinterpret_cast<void*>(m_alloc_ptr + 1); }

  /* User's memory begins at the end of the header */
  size_t size() const { return m_alloc_size - sizeof(SharedAllocationHeader); }

  /* Cannot be 'constexpr' because 'm_count' is volatile */
  int use_count() const { return *static_cast<const volatile int*>(&m_count); }

#ifdef KOKKOS_IMPL_ENABLE_OVERLOAD_HOST_DEVICE
  /* Device tracking_enabled -- always disabled */
  KOKKOS_IMPL_DEVICE_FUNCTION
  static void increment(SharedAllocationRecord*){};
#endif

  /* Increment use count */
  KOKKOS_IMPL_HOST_FUNCTION
  static void increment(SharedAllocationRecord*);

#ifdef KOKKOS_IMPL_ENABLE_OVERLOAD_HOST_DEVICE
  /* Device tracking_enabled -- always disabled */
  KOKKOS_IMPL_DEVICE_FUNCTION
  static void decrement(SharedAllocationRecord*){};
#endif

  /* Decrement use count. If 1->0 then remove from the tracking list and invoke
   * m_dealloc */
  KOKKOS_IMPL_HOST_FUNCTION
  static SharedAllocationRecord* decrement(SharedAllocationRecord*);

  /* Given a root record and data pointer find the record */
  static SharedAllocationRecord* find(SharedAllocationRecord* const,
                                      void* const);

  /*  Sanity check for the whole set of records to which the input record
   * belongs. Locks the set's insert/erase operations until the sanity check is
   * complete.
   */
  static bool is_sane(SharedAllocationRecord*);

  /*  Print host-accessible records */
  static void print_host_accessible_records(
      std::ostream&, const char* const space_name,
      const SharedAllocationRecord* const root, const bool detail);
};

int main()
{
    SharedAllocationHeader* high = new SharedAllocationHeader;
    std::cout << high ->label() << std::endl;
    name *ptr = new name;
    std::cout << ptr -> number << std::endl;
    return 0;
}