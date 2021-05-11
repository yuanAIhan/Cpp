include
bits
shared_ptr_base.h
Go to the documentation of this file.
00001 // shared_ptr and weak_ptr implementation details -*- C++ -*-
00002 
00003 // Copyright (C) 2007, 2008, 2009, 2010, 2011 Free Software Foundation, Inc.
00004 //
00005 // This file is part of the GNU ISO C++ Library.  This library is free
00006 // software; you can redistribute it and/or modify it under the
00007 // terms of the GNU General Public License as published by the
00008 // Free Software Foundation; either version 3, or (at your option)
00009 // any later version.
00010 
00011 // This library is distributed in the hope that it will be useful,
00012 // but WITHOUT ANY WARRANTY; without even the implied warranty of
00013 // MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
00014 // GNU General Public License for more details.
00015 
00016 // Under Section 7 of GPL version 3, you are granted additional
00017 // permissions described in the GCC Runtime Library Exception, version
00018 // 3.1, as published by the Free Software Foundation.
00019 
00020 // You should have received a copy of the GNU General Public License and
00021 // a copy of the GCC Runtime Library Exception along with this program;
00022 // see the files COPYING3 and COPYING.RUNTIME respectively.  If not, see
00023 // <http://www.gnu.org/licenses/>.
00024 
00025 // GCC Note: Based on files from version 1.32.0 of the Boost library.
00026 
00027 //  shared_count.hpp
00028 //  Copyright (c) 2001, 2002, 2003 Peter Dimov and Multi Media Ltd.
00029 
00030 //  shared_ptr.hpp
00031 //  Copyright (C) 1998, 1999 Greg Colvin and Beman Dawes.
00032 //  Copyright (C) 2001, 2002, 2003 Peter Dimov
00033 
00034 //  weak_ptr.hpp
00035 //  Copyright (C) 2001, 2002, 2003 Peter Dimov
00036 
00037 //  enable_shared_from_this.hpp
00038 //  Copyright (C) 2002 Peter Dimov
00039 
00040 // Distributed under the Boost Software License, Version 1.0. (See
00041 // accompanying file LICENSE_1_0.txt or copy at
00042 // http://www.boost.org/LICENSE_1_0.txt)
00043 
00044 /** @file bits/shared_ptr_base.h
00045  *  This is an internal header file, included by other library headers.
00046  *  Do not attempt to use it directly. @headername{memory}
00047  */
00048 
00049 #ifndef _SHARED_PTR_BASE_H
00050 #define _SHARED_PTR_BASE_H 1
00051 
00052 namespace std _GLIBCXX_VISIBILITY(default)
00053 {
00054 _GLIBCXX_BEGIN_NAMESPACE_VERSION
00055 
00056  /**
00057    *  @brief  Exception possibly thrown by @c shared_ptr.
00058    *  @ingroup exceptions
00059    */
00060   class bad_weak_ptr : public std::exception
00061   {
00062   public:
00063     virtual char const*
00064     what() const throw();
00065 
00066     virtual ~bad_weak_ptr() throw();    
00067   };
00068 
00069   // Substitute for bad_weak_ptr object in the case of -fno-exceptions.
00070   inline void
00071   __throw_bad_weak_ptr()
00072   {
00073 #if __EXCEPTIONS
00074     throw bad_weak_ptr();
00075 #else
00076     __builtin_abort();
00077 #endif
00078   }
00079 
00080   using __gnu_cxx::_Lock_policy;
00081   using __gnu_cxx::__default_lock_policy;
00082   using __gnu_cxx::_S_single;
00083   using __gnu_cxx::_S_mutex;
00084   using __gnu_cxx::_S_atomic;
00085 
00086   // Empty helper class except when the template argument is _S_mutex.
00087   template<_Lock_policy _Lp>
00088     class _Mutex_base
00089     {
00090     protected:
00091       // The atomic policy uses fully-fenced builtins, single doesn't care.
00092       enum { _S_need_barriers = 0 };
00093     };
00094 
00095   template<>
00096     class _Mutex_base<_S_mutex>
00097     : public __gnu_cxx::__mutex
00098     {
00099     protected:
00100       // This policy is used when atomic builtins are not available.
00101       // The replacement atomic operations might not have the necessary
00102       // memory barriers.
00103       enum { _S_need_barriers = 1 };
00104     };
00105 
00106   template<_Lock_policy _Lp = __default_lock_policy>
00107     class _Sp_counted_base
00108     : public _Mutex_base<_Lp>
00109     {
00110     public:  
00111       _Sp_counted_base()
00112       : _M_use_count(1), _M_weak_count(1) { }
00113       
00114       virtual
00115       ~_Sp_counted_base() // nothrow 
00116       { }
00117   
00118       // Called when _M_use_count drops to zero, to release the resources
00119       // managed by *this.
00120       virtual void
00121       _M_dispose() = 0; // nothrow
00122       
00123       // Called when _M_weak_count drops to zero.
00124       virtual void
00125       _M_destroy() // nothrow
00126       { delete this; }
00127       
00128       virtual void*
00129       _M_get_deleter(const std::type_info&) = 0;
00130 
00131       void
00132       _M_add_ref_copy()
00133       { __gnu_cxx::__atomic_add_dispatch(&_M_use_count, 1); }
00134   
00135       void
00136       _M_add_ref_lock();
00137       
00138       void
00139       _M_release() // nothrow
00140       {
00141         // Be race-detector-friendly.  For more info see bits/c++config.
00142         _GLIBCXX_SYNCHRONIZATION_HAPPENS_BEFORE(&_M_use_count);
00143     if (__gnu_cxx::__exchange_and_add_dispatch(&_M_use_count, -1) == 1)
00144       {
00145             _GLIBCXX_SYNCHRONIZATION_HAPPENS_AFTER(&_M_use_count);
00146         _M_dispose();
00147         // There must be a memory barrier between dispose() and destroy()
00148         // to ensure that the effects of dispose() are observed in the
00149         // thread that runs destroy().
00150         // See http://gcc.gnu.org/ml/libstdc++/2005-11/msg00136.html
00151         if (_Mutex_base<_Lp>::_S_need_barriers)
00152           {
00153             _GLIBCXX_READ_MEM_BARRIER;
00154             _GLIBCXX_WRITE_MEM_BARRIER;
00155           }
00156 
00157             // Be race-detector-friendly.  For more info see bits/c++config.
00158             _GLIBCXX_SYNCHRONIZATION_HAPPENS_BEFORE(&_M_weak_count);
00159         if (__gnu_cxx::__exchange_and_add_dispatch(&_M_weak_count,
00160                                -1) == 1)
00161               {
00162                 _GLIBCXX_SYNCHRONIZATION_HAPPENS_AFTER(&_M_weak_count);
00163             _M_destroy();
00164               }
00165       }
00166       }
00167   
00168       void
00169       _M_weak_add_ref() // nothrow
00170       { __gnu_cxx::__atomic_add_dispatch(&_M_weak_count, 1); }
00171 
00172       void
00173       _M_weak_release() // nothrow
00174       {
00175         // Be race-detector-friendly. For more info see bits/c++config.
00176         _GLIBCXX_SYNCHRONIZATION_HAPPENS_BEFORE(&_M_weak_count);
00177     if (__gnu_cxx::__exchange_and_add_dispatch(&_M_weak_count, -1) == 1)
00178       {
00179             _GLIBCXX_SYNCHRONIZATION_HAPPENS_AFTER(&_M_weak_count);
00180         if (_Mutex_base<_Lp>::_S_need_barriers)
00181           {
00182             // See _M_release(),
00183             // destroy() must observe results of dispose()
00184             _GLIBCXX_READ_MEM_BARRIER;
00185             _GLIBCXX_WRITE_MEM_BARRIER;
00186           }
00187         _M_destroy();
00188       }
00189       }
00190   
00191       long
00192       _M_get_use_count() const // nothrow
00193       {
00194         // No memory barrier is used here so there is no synchronization
00195         // with other threads.
00196         return const_cast<const volatile _Atomic_word&>(_M_use_count);
00197       }
00198 
00199     private:  
00200       _Sp_counted_base(_Sp_counted_base const&);
00201       _Sp_counted_base& operator=(_Sp_counted_base const&);
00202 
00203       _Atomic_word  _M_use_count;     // #shared
00204       _Atomic_word  _M_weak_count;    // #weak + (#shared != 0)
00205     };
00206 
00207   template<>
00208     inline void
00209     _Sp_counted_base<_S_single>::
00210     _M_add_ref_lock()
00211     {
00212       if (__gnu_cxx::__exchange_and_add_dispatch(&_M_use_count, 1) == 0)
00213     {
00214       _M_use_count = 0;
00215       __throw_bad_weak_ptr();
00216     }
00217     }
00218 
00219   template<>
00220     inline void
00221     _Sp_counted_base<_S_mutex>::
00222     _M_add_ref_lock()
00223     {
00224       __gnu_cxx::__scoped_lock sentry(*this);
00225       if (__gnu_cxx::__exchange_and_add_dispatch(&_M_use_count, 1) == 0)
00226     {
00227       _M_use_count = 0;
00228       __throw_bad_weak_ptr();
00229     }
00230     }
00231 
00232   template<> 
00233     inline void
00234     _Sp_counted_base<_S_atomic>::
00235     _M_add_ref_lock()
00236     {
00237       // Perform lock-free add-if-not-zero operation.
00238       _Atomic_word __count;
00239       do
00240     {
00241       __count = _M_use_count;
00242       if (__count == 0)
00243         __throw_bad_weak_ptr();
00244       
00245       // Replace the current counter value with the old value + 1, as
00246       // long as it's not changed meanwhile. 
00247     }
00248       while (!__sync_bool_compare_and_swap(&_M_use_count, __count,
00249                        __count + 1));
00250     }
00251 
00252 
00253   // Forward declarations.
00254   template<typename _Tp, _Lock_policy _Lp = __default_lock_policy>
00255     class __shared_ptr;
00256 
00257   template<typename _Tp, _Lock_policy _Lp = __default_lock_policy>
00258     class __weak_ptr;
00259 
00260   template<typename _Tp, _Lock_policy _Lp = __default_lock_policy>
00261     class __enable_shared_from_this;
00262 
00263   template<typename _Tp>
00264     class shared_ptr;
00265 
00266   template<typename _Tp>
00267     class weak_ptr;
00268 
00269   template<typename _Tp>
00270     struct owner_less;
00271 
00272   template<typename _Tp>
00273     class enable_shared_from_this;
00274 
00275   template<_Lock_policy _Lp = __default_lock_policy>
00276     class __weak_count;
00277 
00278   template<_Lock_policy _Lp = __default_lock_policy>
00279     class __shared_count;
00280 
00281 
00282   // Counted ptr with no deleter or allocator support
00283   template<typename _Ptr, _Lock_policy _Lp>
00284     class _Sp_counted_ptr : public _Sp_counted_base<_Lp>
00285     {
00286     public:
00287       explicit
00288       _Sp_counted_ptr(_Ptr __p)
00289       : _M_ptr(__p) { }
00290 
00291       virtual void
00292       _M_dispose() // nothrow
00293       { delete _M_ptr; }
00294 
00295       virtual void
00296       _M_destroy() // nothrow
00297       { delete this; }
00298 
00299       virtual void*
00300       _M_get_deleter(const std::type_info&)
00301       { return 0; }
00302 
00303       _Sp_counted_ptr(const _Sp_counted_ptr&) = delete;
00304       _Sp_counted_ptr& operator=(const _Sp_counted_ptr&) = delete;
00305 
00306     protected:
00307       _Ptr             _M_ptr;  // copy constructor must not throw
00308     };
00309 
00310   template<>
00311     inline void
00312     _Sp_counted_ptr<nullptr_t, _S_single>::_M_dispose() { }
00313 
00314   template<>
00315     inline void
00316     _Sp_counted_ptr<nullptr_t, _S_mutex>::_M_dispose() { }
00317 
00318   template<>
00319     inline void
00320     _Sp_counted_ptr<nullptr_t, _S_atomic>::_M_dispose() { }
00321 
00322   // Support for custom deleter and/or allocator
00323   template<typename _Ptr, typename _Deleter, typename _Alloc, _Lock_policy _Lp>
00324     class _Sp_counted_deleter : public _Sp_counted_base<_Lp>
00325     {
00326       typedef typename _Alloc::template
00327       rebind<_Sp_counted_deleter>::other _My_alloc_type;
00328 
00329       // Helper class that stores the Deleter and also acts as an allocator.
00330       // Used to dispose of the owned pointer and the internal refcount
00331       // Requires that copies of _Alloc can free each other's memory.
00332       struct _My_Deleter
00333       : public _My_alloc_type    // copy constructor must not throw
00334       {
00335     _Deleter _M_del;         // copy constructor must not throw
00336     _My_Deleter(_Deleter __d, const _Alloc& __a)
00337     : _My_alloc_type(__a), _M_del(__d) { }
00338       };
00339 
00340     public:
00341       // __d(__p) must not throw.
00342       _Sp_counted_deleter(_Ptr __p, _Deleter __d)
00343       : _M_ptr(__p), _M_del(__d, _Alloc()) { }
00344 
00345       // __d(__p) must not throw.
00346       _Sp_counted_deleter(_Ptr __p, _Deleter __d, const _Alloc& __a)
00347       : _M_ptr(__p), _M_del(__d, __a) { }
00348 
00349       virtual void
00350       _M_dispose() // nothrow
00351       { _M_del._M_del(_M_ptr); }
00352 
00353       virtual void
00354       _M_destroy() // nothrow
00355       {
00356     _My_alloc_type __a(_M_del);
00357     this->~_Sp_counted_deleter();
00358     __a.deallocate(this, 1);
00359       }
00360 
00361       virtual void*
00362       _M_get_deleter(const std::type_info& __ti)
00363       {
00364 #ifdef __GXX_RTTI
00365         return __ti == typeid(_Deleter) ? &_M_del._M_del : 0;
00366 #else
00367         return 0;
00368 #endif
00369       }
00370 
00371     protected:
00372       _Ptr             _M_ptr;  // copy constructor must not throw
00373       _My_Deleter      _M_del;  // copy constructor must not throw
00374     };
00375 
00376   // helpers for make_shared / allocate_shared
00377 
00378   template<typename _Tp>
00379     struct _Sp_destroy_inplace
00380     {
00381       void operator()(_Tp* __p) const { if (__p) __p->~_Tp(); }
00382     };
00383 
00384   struct _Sp_make_shared_tag { };
00385 
00386   template<typename _Tp, typename _Alloc, _Lock_policy _Lp>
00387     class _Sp_counted_ptr_inplace
00388     : public _Sp_counted_deleter<_Tp*, _Sp_destroy_inplace<_Tp>, _Alloc, _Lp>
00389     {
00390       typedef _Sp_counted_deleter<_Tp*, _Sp_destroy_inplace<_Tp>, _Alloc, _Lp>
00391     _Base_type;
00392 
00393     public:
00394       explicit
00395       _Sp_counted_ptr_inplace(_Alloc __a)
00396       : _Base_type(static_cast<_Tp*>(0), _Sp_destroy_inplace<_Tp>(), __a)
00397       , _M_storage()
00398       {
00399     void* __p = &_M_storage;
00400     ::new (__p) _Tp();  // might throw
00401     _Base_type::_M_ptr = static_cast<_Tp*>(__p);
00402       }
00403 
00404       template<typename... _Args>
00405     _Sp_counted_ptr_inplace(_Alloc __a, _Args&&... __args)
00406     : _Base_type(static_cast<_Tp*>(0), _Sp_destroy_inplace<_Tp>(), __a)
00407     , _M_storage()
00408     {
00409       void* __p = &_M_storage;
00410       ::new (__p) _Tp(std::forward<_Args>(__args)...);  // might throw
00411       _Base_type::_M_ptr = static_cast<_Tp*>(__p);
00412     }
00413 
00414       // Override because the allocator needs to know the dynamic type
00415       virtual void
00416       _M_destroy() // nothrow
00417       {
00418     typedef typename _Alloc::template
00419         rebind<_Sp_counted_ptr_inplace>::other _My_alloc_type;
00420     _My_alloc_type __a(_Base_type::_M_del);
00421     this->~_Sp_counted_ptr_inplace();
00422     __a.deallocate(this, 1);
00423       }
00424 
00425       // Sneaky trick so __shared_ptr can get the managed pointer
00426       virtual void*
00427       _M_get_deleter(const std::type_info& __ti)
00428       {
00429 #ifdef __GXX_RTTI
00430     return __ti == typeid(_Sp_make_shared_tag)
00431            ? static_cast<void*>(&_M_storage)
00432            : _Base_type::_M_get_deleter(__ti);
00433 #else
00434         return 0;
00435 #endif
00436       }
00437 
00438     private:
00439       typename aligned_storage<sizeof(_Tp), alignment_of<_Tp>::value>::type
00440     _M_storage;
00441     };
00442 
00443   template<_Lock_policy _Lp>
00444     class __shared_count
00445     {
00446     public:
00447       constexpr __shared_count() : _M_pi(0) // nothrow
00448       { }
00449 
00450       template<typename _Ptr>
00451         explicit
00452     __shared_count(_Ptr __p) : _M_pi(0)
00453     {
00454       __try
00455         {
00456           _M_pi = new _Sp_counted_ptr<_Ptr, _Lp>(__p);
00457         }
00458       __catch(...)
00459         {
00460           delete __p;
00461           __throw_exception_again;
00462         }
00463     }
00464 
00465       template<typename _Ptr, typename _Deleter>
00466     __shared_count(_Ptr __p, _Deleter __d) : _M_pi(0)
00467     {
00468       // The allocator's value_type doesn't matter, will rebind it anyway.
00469       typedef std::allocator<int> _Alloc;
00470       typedef _Sp_counted_deleter<_Ptr, _Deleter, _Alloc, _Lp> _Sp_cd_type;
00471       typedef std::allocator<_Sp_cd_type> _Alloc2;
00472       _Alloc2 __a2;
00473       __try
00474         {
00475           _M_pi = __a2.allocate(1);
00476           ::new(static_cast<void*>(_M_pi)) _Sp_cd_type(__p, __d);
00477         }
00478       __catch(...)
00479         {
00480           __d(__p); // Call _Deleter on __p.
00481           if (_M_pi)
00482         __a2.deallocate(static_cast<_Sp_cd_type*>(_M_pi), 1);
00483           __throw_exception_again;
00484         }
00485     }
00486 
00487       template<typename _Ptr, typename _Deleter, typename _Alloc>
00488     __shared_count(_Ptr __p, _Deleter __d, _Alloc __a) : _M_pi(0)
00489     {
00490       typedef _Sp_counted_deleter<_Ptr, _Deleter, _Alloc, _Lp> _Sp_cd_type;
00491       typedef typename _Alloc::template rebind<_Sp_cd_type>::other _Alloc2;
00492       _Alloc2 __a2(__a);
00493       __try
00494         {
00495           _M_pi = __a2.allocate(1);
00496           ::new(static_cast<void*>(_M_pi)) _Sp_cd_type(__p, __d, __a);
00497         }
00498       __catch(...)
00499         {
00500           __d(__p); // Call _Deleter on __p.
00501           if (_M_pi)
00502         __a2.deallocate(static_cast<_Sp_cd_type*>(_M_pi), 1);
00503           __throw_exception_again;
00504         }
00505     }
00506 
00507       template<typename _Tp, typename _Alloc, typename... _Args>
00508     __shared_count(_Sp_make_shared_tag, _Tp*, const _Alloc& __a,
00509                _Args&&... __args)
00510     : _M_pi(0)
00511     {
00512       typedef _Sp_counted_ptr_inplace<_Tp, _Alloc, _Lp> _Sp_cp_type;
00513       typedef typename _Alloc::template rebind<_Sp_cp_type>::other _Alloc2;
00514       _Alloc2 __a2(__a);
00515       __try
00516         {
00517           _M_pi = __a2.allocate(1);
00518           ::new(static_cast<void*>(_M_pi)) _Sp_cp_type(__a,
00519             std::forward<_Args>(__args)...);
00520         }
00521       __catch(...)
00522         {
00523           if (_M_pi)
00524         __a2.deallocate(static_cast<_Sp_cp_type*>(_M_pi), 1);
00525           __throw_exception_again;
00526         }
00527     }
00528 
00529 #if _GLIBCXX_USE_DEPRECATED
00530       // Special case for auto_ptr<_Tp> to provide the strong guarantee.
00531       template<typename _Tp>
00532         explicit
00533     __shared_count(std::auto_ptr<_Tp>&& __r)
00534     : _M_pi(new _Sp_counted_ptr<_Tp*, _Lp>(__r.get()))
00535     { __r.release(); }
00536 #endif
00537 
00538       // Special case for unique_ptr<_Tp,_Del> to provide the strong guarantee.
00539       template<typename _Tp, typename _Del>
00540         explicit
00541     __shared_count(std::unique_ptr<_Tp, _Del>&& __r)
00542     : _M_pi(_S_create_from_up(std::move(__r)))
00543     { __r.release(); }
00544 
00545       // Throw bad_weak_ptr when __r._M_get_use_count() == 0.
00546       explicit __shared_count(const __weak_count<_Lp>& __r);
00547 
00548       ~__shared_count() // nothrow
00549       {
00550     if (_M_pi != 0)
00551       _M_pi->_M_release();
00552       }
00553 
00554       __shared_count(const __shared_count& __r)
00555       : _M_pi(__r._M_pi) // nothrow
00556       {
00557     if (_M_pi != 0)
00558       _M_pi->_M_add_ref_copy();
00559       }
00560 
00561       __shared_count&
00562       operator=(const __shared_count& __r) // nothrow
00563       {
00564     _Sp_counted_base<_Lp>* __tmp = __r._M_pi;
00565     if (__tmp != _M_pi)
00566       {
00567         if (__tmp != 0)
00568           __tmp->_M_add_ref_copy();
00569         if (_M_pi != 0)
00570           _M_pi->_M_release();
00571         _M_pi = __tmp;
00572       }
00573     return *this;
00574       }
00575 
00576       void
00577       _M_swap(__shared_count& __r) // nothrow
00578       {
00579     _Sp_counted_base<_Lp>* __tmp = __r._M_pi;
00580     __r._M_pi = _M_pi;
00581     _M_pi = __tmp;
00582       }
00583 
00584       long
00585       _M_get_use_count() const // nothrow
00586       { return _M_pi != 0 ? _M_pi->_M_get_use_count() : 0; }
00587 
00588       bool
00589       _M_unique() const // nothrow
00590       { return this->_M_get_use_count() == 1; }
00591 
00592       void*
00593       _M_get_deleter(const std::type_info& __ti) const
00594       { return _M_pi ? _M_pi->_M_get_deleter(__ti) : 0; }
00595 
00596       bool
00597       _M_less(const __shared_count& __rhs) const
00598       { return std::less<_Sp_counted_base<_Lp>*>()(this->_M_pi, __rhs._M_pi); }
00599 
00600       bool
00601       _M_less(const __weak_count<_Lp>& __rhs) const
00602       { return std::less<_Sp_counted_base<_Lp>*>()(this->_M_pi, __rhs._M_pi); }
00603 
00604       // Friend function injected into enclosing namespace and found by ADL
00605       friend inline bool
00606       operator==(const __shared_count& __a, const __shared_count& __b)
00607       { return __a._M_pi == __b._M_pi; }
00608 
00609     private:
00610       friend class __weak_count<_Lp>;
00611 
00612       template<typename _Tp, typename _Del>
00613     static _Sp_counted_base<_Lp>*
00614     _S_create_from_up(std::unique_ptr<_Tp, _Del>&& __r,
00615       typename std::enable_if<!std::is_reference<_Del>::value>::type* = 0)
00616     {
00617       return new _Sp_counted_deleter<_Tp*, _Del, std::allocator<_Tp>,
00618         _Lp>(__r.get(), __r.get_deleter());
00619     }
00620 
00621       template<typename _Tp, typename _Del>
00622     static _Sp_counted_base<_Lp>*
00623     _S_create_from_up(std::unique_ptr<_Tp, _Del>&& __r,
00624       typename std::enable_if<std::is_reference<_Del>::value>::type* = 0)
00625     {
00626       typedef typename std::remove_reference<_Del>::type _Del1;
00627       typedef std::reference_wrapper<_Del1> _Del2;
00628       return new _Sp_counted_deleter<_Tp*, _Del2, std::allocator<_Tp>,
00629         _Lp>(__r.get(), std::ref(__r.get_deleter()));
00630     }
00631 
00632       _Sp_counted_base<_Lp>*  _M_pi;
00633     };
00634 
00635 
00636   template<_Lock_policy _Lp>
00637     class __weak_count
00638     {
00639     public:
00640       constexpr __weak_count() : _M_pi(0) // nothrow
00641       { }
00642 
00643       __weak_count(const __shared_count<_Lp>& __r) : _M_pi(__r._M_pi) // nothrow
00644       {
00645     if (_M_pi != 0)
00646       _M_pi->_M_weak_add_ref();
00647       }
00648 
00649       __weak_count(const __weak_count<_Lp>& __r) : _M_pi(__r._M_pi) // nothrow
00650       {
00651     if (_M_pi != 0)
00652       _M_pi->_M_weak_add_ref();
00653       }
00654 
00655       ~__weak_count() // nothrow
00656       {
00657     if (_M_pi != 0)
00658       _M_pi->_M_weak_release();
00659       }
00660 
00661       __weak_count<_Lp>&
00662       operator=(const __shared_count<_Lp>& __r) // nothrow
00663       {
00664     _Sp_counted_base<_Lp>* __tmp = __r._M_pi;
00665     if (__tmp != 0)
00666       __tmp->_M_weak_add_ref();
00667     if (_M_pi != 0)
00668       _M_pi->_M_weak_release();
00669     _M_pi = __tmp;
00670     return *this;
00671       }
00672 
00673       __weak_count<_Lp>&
00674       operator=(const __weak_count<_Lp>& __r) // nothrow
00675       {
00676     _Sp_counted_base<_Lp>* __tmp = __r._M_pi;
00677     if (__tmp != 0)
00678       __tmp->_M_weak_add_ref();
00679     if (_M_pi != 0)
00680       _M_pi->_M_weak_release();
00681     _M_pi = __tmp;
00682     return *this;
00683       }
00684 
00685       void
00686       _M_swap(__weak_count<_Lp>& __r) // nothrow
00687       {
00688     _Sp_counted_base<_Lp>* __tmp = __r._M_pi;
00689     __r._M_pi = _M_pi;
00690     _M_pi = __tmp;
00691       }
00692 
00693       long
00694       _M_get_use_count() const // nothrow
00695       { return _M_pi != 0 ? _M_pi->_M_get_use_count() : 0; }
00696 
00697       bool
00698       _M_less(const __weak_count& __rhs) const
00699       { return std::less<_Sp_counted_base<_Lp>*>()(this->_M_pi, __rhs._M_pi); }
00700 
00701       bool
00702       _M_less(const __shared_count<_Lp>& __rhs) const
00703       { return std::less<_Sp_counted_base<_Lp>*>()(this->_M_pi, __rhs._M_pi); }
00704 
00705       // Friend function injected into enclosing namespace and found by ADL
00706       friend inline bool
00707       operator==(const __weak_count& __a, const __weak_count& __b)
00708       { return __a._M_pi == __b._M_pi; }
00709 
00710     private:
00711       friend class __shared_count<_Lp>;
00712 
00713       _Sp_counted_base<_Lp>*  _M_pi;
00714     };
00715 
00716   // Now that __weak_count is defined we can define this constructor:
00717   template<_Lock_policy _Lp>
00718     inline __shared_count<_Lp>:: __shared_count(const __weak_count<_Lp>& __r)
00719     : _M_pi(__r._M_pi)
00720     {
00721       if (_M_pi != 0)
00722     _M_pi->_M_add_ref_lock();
00723       else
00724     __throw_bad_weak_ptr();
00725     }
00726 
00727 
00728   // Support for enable_shared_from_this.
00729 
00730   // Friend of __enable_shared_from_this.
00731   template<_Lock_policy _Lp, typename _Tp1, typename _Tp2>
00732     void
00733     __enable_shared_from_this_helper(const __shared_count<_Lp>&,
00734                      const __enable_shared_from_this<_Tp1,
00735                      _Lp>*, const _Tp2*);
00736 
00737   // Friend of enable_shared_from_this.
00738   template<typename _Tp1, typename _Tp2>
00739     void
00740     __enable_shared_from_this_helper(const __shared_count<>&,
00741                      const enable_shared_from_this<_Tp1>*,
00742                      const _Tp2*);
00743 
00744   template<_Lock_policy _Lp>
00745     inline void
00746     __enable_shared_from_this_helper(const __shared_count<_Lp>&, ...)
00747     { }
00748 
00749 
00750   template<typename _Tp, _Lock_policy _Lp>
00751     class __shared_ptr
00752     {
00753     public:
00754       typedef _Tp   element_type;
00755 
00756       constexpr __shared_ptr()
00757       : _M_ptr(0), _M_refcount() // never throws
00758       { }
00759 
00760       template<typename _Tp1>
00761     explicit __shared_ptr(_Tp1* __p)
00762         : _M_ptr(__p), _M_refcount(__p)
00763     {
00764       __glibcxx_function_requires(_ConvertibleConcept<_Tp1*, _Tp*>)
00765       static_assert( sizeof(_Tp1) > 0, "incomplete type" );
00766       __enable_shared_from_this_helper(_M_refcount, __p, __p);
00767     }
00768 
00769       template<typename _Tp1, typename _Deleter>
00770     __shared_ptr(_Tp1* __p, _Deleter __d)
00771     : _M_ptr(__p), _M_refcount(__p, __d)
00772     {
00773       __glibcxx_function_requires(_ConvertibleConcept<_Tp1*, _Tp*>)
00774       // TODO requires _Deleter CopyConstructible and __d(__p) well-formed
00775       __enable_shared_from_this_helper(_M_refcount, __p, __p);
00776     }
00777 
00778       template<typename _Tp1, typename _Deleter, typename _Alloc>
00779     __shared_ptr(_Tp1* __p, _Deleter __d, _Alloc __a)
00780     : _M_ptr(__p), _M_refcount(__p, __d, std::move(__a))
00781     {
00782       __glibcxx_function_requires(_ConvertibleConcept<_Tp1*, _Tp*>)
00783       // TODO requires _Deleter CopyConstructible and __d(__p) well-formed
00784       __enable_shared_from_this_helper(_M_refcount, __p, __p);
00785     }
00786 
00787       template<typename _Deleter>
00788     __shared_ptr(nullptr_t __p, _Deleter __d)
00789     : _M_ptr(0), _M_refcount(__p, __d)
00790     { }
00791 
00792       template<typename _Deleter, typename _Alloc>
00793         __shared_ptr(nullptr_t __p, _Deleter __d, _Alloc __a)
00794     : _M_ptr(0), _M_refcount(__p, __d, std::move(__a))
00795     { }
00796 
00797       template<typename _Tp1>
00798     __shared_ptr(const __shared_ptr<_Tp1, _Lp>& __r, _Tp* __p)
00799     : _M_ptr(__p), _M_refcount(__r._M_refcount) // never throws
00800     { }
00801 
00802       //  generated copy constructor, assignment, destructor are fine.
00803 
00804       template<typename _Tp1, typename = typename
00805            std::enable_if<std::is_convertible<_Tp1*, _Tp*>::value>::type>
00806     __shared_ptr(const __shared_ptr<_Tp1, _Lp>& __r)
00807     : _M_ptr(__r._M_ptr), _M_refcount(__r._M_refcount) // never throws
00808     { }
00809 
00810       __shared_ptr(__shared_ptr&& __r)
00811       : _M_ptr(__r._M_ptr), _M_refcount() // never throws
00812       {
00813     _M_refcount._M_swap(__r._M_refcount);
00814     __r._M_ptr = 0;
00815       }
00816 
00817       template<typename _Tp1, typename = typename
00818            std::enable_if<std::is_convertible<_Tp1*, _Tp*>::value>::type>
00819     __shared_ptr(__shared_ptr<_Tp1, _Lp>&& __r)
00820     : _M_ptr(__r._M_ptr), _M_refcount() // never throws
00821     {
00822       _M_refcount._M_swap(__r._M_refcount);
00823       __r._M_ptr = 0;
00824     }
00825 
00826       template<typename _Tp1>
00827     explicit __shared_ptr(const __weak_ptr<_Tp1, _Lp>& __r)
00828     : _M_refcount(__r._M_refcount) // may throw
00829     {
00830       __glibcxx_function_requires(_ConvertibleConcept<_Tp1*, _Tp*>)
00831 
00832       // It is now safe to copy __r._M_ptr, as
00833       // _M_refcount(__r._M_refcount) did not throw.
00834       _M_ptr = __r._M_ptr;
00835     }
00836 
00837       // If an exception is thrown this constructor has no effect.
00838       template<typename _Tp1, typename _Del>
00839     __shared_ptr(std::unique_ptr<_Tp1, _Del>&& __r)
00840     : _M_ptr(__r.get()), _M_refcount()
00841     {
00842       __glibcxx_function_requires(_ConvertibleConcept<_Tp1*, _Tp*>)
00843       _Tp1* __tmp = __r.get();
00844       _M_refcount = __shared_count<_Lp>(std::move(__r));
00845       __enable_shared_from_this_helper(_M_refcount, __tmp, __tmp);
00846     }
00847 
00848 #if _GLIBCXX_USE_DEPRECATED
00849       // Postcondition: use_count() == 1 and __r.get() == 0
00850       template<typename _Tp1>
00851     __shared_ptr(std::auto_ptr<_Tp1>&& __r)
00852     : _M_ptr(__r.get()), _M_refcount()
00853     {
00854       __glibcxx_function_requires(_ConvertibleConcept<_Tp1*, _Tp*>)
00855       static_assert( sizeof(_Tp1) > 0, "incomplete type" );
00856       _Tp1* __tmp = __r.get();
00857       _M_refcount = __shared_count<_Lp>(std::move(__r));
00858       __enable_shared_from_this_helper(_M_refcount, __tmp, __tmp);
00859     }
00860 #endif
00861 
00862       /* TODO: use delegating constructor */
00863       constexpr __shared_ptr(nullptr_t)
00864       : _M_ptr(0), _M_refcount() // never throws
00865       { }
00866 
00867       template<typename _Tp1>
00868     __shared_ptr&
00869     operator=(const __shared_ptr<_Tp1, _Lp>& __r) // never throws
00870     {
00871       _M_ptr = __r._M_ptr;
00872       _M_refcount = __r._M_refcount; // __shared_count::op= doesn't throw
00873       return *this;
00874     }
00875 
00876 #if _GLIBCXX_USE_DEPRECATED
00877       template<typename _Tp1>
00878     __shared_ptr&
00879     operator=(std::auto_ptr<_Tp1>&& __r)
00880     {
00881       __shared_ptr(std::move(__r)).swap(*this);
00882       return *this;
00883     }
00884 #endif
00885 
00886       __shared_ptr&
00887       operator=(__shared_ptr&& __r)
00888       {
00889     __shared_ptr(std::move(__r)).swap(*this);
00890     return *this;
00891       }
00892 
00893       template<class _Tp1>
00894     __shared_ptr&
00895     operator=(__shared_ptr<_Tp1, _Lp>&& __r)
00896     {
00897       __shared_ptr(std::move(__r)).swap(*this);
00898       return *this;
00899     }
00900 
00901       template<typename _Tp1, typename _Del>
00902     __shared_ptr&
00903     operator=(std::unique_ptr<_Tp1, _Del>&& __r)
00904     {
00905       __shared_ptr(std::move(__r)).swap(*this);
00906       return *this;
00907     }
00908 
00909       void
00910       reset() // never throws
00911       { __shared_ptr().swap(*this); }
00912 
00913       template<typename _Tp1>
00914     void
00915     reset(_Tp1* __p) // _Tp1 must be complete.
00916     {
00917       // Catch self-reset errors.
00918       _GLIBCXX_DEBUG_ASSERT(__p == 0 || __p != _M_ptr);
00919       __shared_ptr(__p).swap(*this);
00920     }
00921 
00922       template<typename _Tp1, typename _Deleter>
00923     void
00924     reset(_Tp1* __p, _Deleter __d)
00925     { __shared_ptr(__p, __d).swap(*this); }
00926 
00927       template<typename _Tp1, typename _Deleter, typename _Alloc>
00928     void
00929         reset(_Tp1* __p, _Deleter __d, _Alloc __a)
00930         { __shared_ptr(__p, __d, std::move(__a)).swap(*this); }
00931 
00932       // Allow class instantiation when _Tp is [cv-qual] void.
00933       typename std::add_lvalue_reference<_Tp>::type
00934       operator*() const // never throws
00935       {
00936     _GLIBCXX_DEBUG_ASSERT(_M_ptr != 0);
00937     return *_M_ptr;
00938       }
00939 
00940       _Tp*
00941       operator->() const // never throws
00942       {
00943     _GLIBCXX_DEBUG_ASSERT(_M_ptr != 0);
00944     return _M_ptr;
00945       }
00946 
00947       _Tp*
00948       get() const // never throws
00949       { return _M_ptr; }
00950 
00951       explicit operator bool() const // never throws
00952       { return _M_ptr == 0 ? false : true; }
00953 
00954       bool
00955       unique() const // never throws
00956       { return _M_refcount._M_unique(); }
00957 
00958       long
00959       use_count() const // never throws
00960       { return _M_refcount._M_get_use_count(); }
00961 
00962       void
00963       swap(__shared_ptr<_Tp, _Lp>& __other) // never throws
00964       {
00965     std::swap(_M_ptr, __other._M_ptr);
00966     _M_refcount._M_swap(__other._M_refcount);
00967       }
00968 
00969       template<typename _Tp1>
00970     bool
00971     owner_before(__shared_ptr<_Tp1, _Lp> const& __rhs) const
00972     { return _M_refcount._M_less(__rhs._M_refcount); }
00973 
00974       template<typename _Tp1>
00975     bool
00976     owner_before(__weak_ptr<_Tp1, _Lp> const& __rhs) const
00977     { return _M_refcount._M_less(__rhs._M_refcount); }
00978 
00979 #ifdef __GXX_RTTI
00980     protected:
00981       // This constructor is non-standard, it is used by allocate_shared.
00982       template<typename _Alloc, typename... _Args>
00983     __shared_ptr(_Sp_make_shared_tag __tag, const _Alloc& __a,
00984              _Args&&... __args)
00985     : _M_ptr(), _M_refcount(__tag, (_Tp*)0, __a,
00986                 std::forward<_Args>(__args)...)
00987     {
00988       // _M_ptr needs to point to the newly constructed object.
00989       // This relies on _Sp_counted_ptr_inplace::_M_get_deleter.
00990       void* __p = _M_refcount._M_get_deleter(typeid(__tag));
00991       _M_ptr = static_cast<_Tp*>(__p);
00992       __enable_shared_from_this_helper(_M_refcount, _M_ptr, _M_ptr);
00993     }
00994 #else
00995       template<typename _Alloc>
00996         struct _Deleter
00997         {
00998           void operator()(_Tp* __ptr)
00999           {
01000             _M_alloc.destroy(__ptr);
01001             _M_alloc.deallocate(__ptr, 1);
01002           }
01003           _Alloc _M_alloc;
01004         };
01005 
01006       template<typename _Alloc, typename... _Args>
01007     __shared_ptr(_Sp_make_shared_tag __tag, const _Alloc& __a,
01008              _Args&&... __args)
01009     : _M_ptr(), _M_refcount()
01010         {
01011       typedef typename _Alloc::template rebind<_Tp>::other _Alloc2;
01012           _Deleter<_Alloc2> __del = { _Alloc2(__a) };
01013           _M_ptr = __del._M_alloc.allocate(1);
01014       __try
01015         {
01016               __del._M_alloc.construct(_M_ptr, std::forward<_Args>(__args)...);
01017         }
01018       __catch(...)
01019         {
01020               __del._M_alloc.deallocate(_M_ptr, 1);
01021           __throw_exception_again;
01022         }
01023           __shared_count<_Lp> __count(_M_ptr, __del, __del._M_alloc);
01024           _M_refcount._M_swap(__count);
01025       __enable_shared_from_this_helper(_M_refcount, _M_ptr, _M_ptr);
01026         }
01027 #endif
01028 
01029       template<typename _Tp1, _Lock_policy _Lp1, typename _Alloc,
01030            typename... _Args>
01031     friend __shared_ptr<_Tp1, _Lp1>
01032     __allocate_shared(const _Alloc& __a, _Args&&... __args);
01033 
01034     private:
01035       void*
01036       _M_get_deleter(const std::type_info& __ti) const
01037       { return _M_refcount._M_get_deleter(__ti); }
01038 
01039       template<typename _Tp1, _Lock_policy _Lp1> friend class __shared_ptr;
01040       template<typename _Tp1, _Lock_policy _Lp1> friend class __weak_ptr;
01041 
01042       template<typename _Del, typename _Tp1, _Lock_policy _Lp1>
01043     friend _Del* get_deleter(const __shared_ptr<_Tp1, _Lp1>&);
01044 
01045       _Tp*         _M_ptr;         // Contained pointer.
01046       __shared_count<_Lp>  _M_refcount;    // Reference counter.
01047     };
01048 
01049 
01050   // 20.8.13.2.7 shared_ptr comparisons
01051   template<typename _Tp1, typename _Tp2, _Lock_policy _Lp>
01052     inline bool
01053     operator==(const __shared_ptr<_Tp1, _Lp>& __a,
01054            const __shared_ptr<_Tp2, _Lp>& __b)
01055     { return __a.get() == __b.get(); }
01056 
01057   template<typename _Tp, _Lock_policy _Lp>
01058     inline bool
01059     operator==(const __shared_ptr<_Tp, _Lp>& __a, nullptr_t)
01060     { return __a.get() == nullptr; }
01061 
01062   template<typename _Tp, _Lock_policy _Lp>
01063     inline bool
01064     operator==(nullptr_t, const __shared_ptr<_Tp, _Lp>& __b)
01065     { return nullptr == __b.get(); }
01066 
01067   template<typename _Tp1, typename _Tp2, _Lock_policy _Lp>
01068     inline bool
01069     operator!=(const __shared_ptr<_Tp1, _Lp>& __a,
01070            const __shared_ptr<_Tp2, _Lp>& __b)
01071     { return __a.get() != __b.get(); }
01072 
01073   template<typename _Tp, _Lock_policy _Lp>
01074     inline bool
01075     operator!=(const __shared_ptr<_Tp, _Lp>& __a, nullptr_t)
01076     { return __a.get() != nullptr; }
01077 
01078   template<typename _Tp, _Lock_policy _Lp>
01079     inline bool
01080     operator!=(nullptr_t, const __shared_ptr<_Tp, _Lp>& __b)
01081     { return nullptr != __b.get(); }
01082 
01083   template<typename _Tp1, typename _Tp2, _Lock_policy _Lp>
01084     inline bool
01085     operator<(const __shared_ptr<_Tp1, _Lp>& __a,
01086           const __shared_ptr<_Tp2, _Lp>& __b)
01087     { return __a.get() < __b.get(); }
01088 
01089   template<typename _Sp>
01090     struct _Sp_less : public binary_function<_Sp, _Sp, bool>
01091     {
01092       bool
01093       operator()(const _Sp& __lhs, const _Sp& __rhs) const
01094       {
01095     typedef typename _Sp::element_type element_type;
01096     return std::less<element_type*>()(__lhs.get(), __rhs.get());
01097       }
01098     };
01099 
01100   template<typename _Tp, _Lock_policy _Lp>
01101     struct less<__shared_ptr<_Tp, _Lp>>
01102     : public _Sp_less<__shared_ptr<_Tp, _Lp>>
01103     { };
01104 
01105   // 2.2.3.8 shared_ptr specialized algorithms.
01106   template<typename _Tp, _Lock_policy _Lp>
01107     inline void
01108     swap(__shared_ptr<_Tp, _Lp>& __a, __shared_ptr<_Tp, _Lp>& __b)
01109     { __a.swap(__b); }
01110 
01111   // 2.2.3.9 shared_ptr casts
01112 
01113   // The seemingly equivalent code:
01114   // shared_ptr<_Tp, _Lp>(static_cast<_Tp*>(__r.get()))
01115   // will eventually result in undefined behaviour, attempting to
01116   // delete the same object twice.
01117   /// static_pointer_cast
01118   template<typename _Tp, typename _Tp1, _Lock_policy _Lp>
01119     inline __shared_ptr<_Tp, _Lp>
01120     static_pointer_cast(const __shared_ptr<_Tp1, _Lp>& __r)
01121     { return __shared_ptr<_Tp, _Lp>(__r, static_cast<_Tp*>(__r.get())); }
01122 
01123   // The seemingly equivalent code:
01124   // shared_ptr<_Tp, _Lp>(const_cast<_Tp*>(__r.get()))
01125   // will eventually result in undefined behaviour, attempting to
01126   // delete the same object twice.
01127   /// const_pointer_cast
01128   template<typename _Tp, typename _Tp1, _Lock_policy _Lp>
01129     inline __shared_ptr<_Tp, _Lp>
01130     const_pointer_cast(const __shared_ptr<_Tp1, _Lp>& __r)
01131     { return __shared_ptr<_Tp, _Lp>(__r, const_cast<_Tp*>(__r.get())); }
01132 
01133   // The seemingly equivalent code:
01134   // shared_ptr<_Tp, _Lp>(dynamic_cast<_Tp*>(__r.get()))
01135   // will eventually result in undefined behaviour, attempting to
01136   // delete the same object twice.
01137   /// dynamic_pointer_cast
01138   template<typename _Tp, typename _Tp1, _Lock_policy _Lp>
01139     inline __shared_ptr<_Tp, _Lp>
01140     dynamic_pointer_cast(const __shared_ptr<_Tp1, _Lp>& __r)
01141     {
01142       if (_Tp* __p = dynamic_cast<_Tp*>(__r.get()))
01143     return __shared_ptr<_Tp, _Lp>(__r, __p);
01144       return __shared_ptr<_Tp, _Lp>();
01145     }
01146 
01147 
01148   template<typename _Tp, _Lock_policy _Lp>
01149     class __weak_ptr
01150     {
01151     public:
01152       typedef _Tp element_type;
01153 
01154       constexpr __weak_ptr()
01155       : _M_ptr(0), _M_refcount() // never throws
01156       { }
01157 
01158       // Generated copy constructor, assignment, destructor are fine.
01159 
01160       // The "obvious" converting constructor implementation:
01161       //
01162       //  template<typename _Tp1>
01163       //    __weak_ptr(const __weak_ptr<_Tp1, _Lp>& __r)
01164       //    : _M_ptr(__r._M_ptr), _M_refcount(__r._M_refcount) // never throws
01165       //    { }
01166       //
01167       // has a serious problem.
01168       //
01169       //  __r._M_ptr may already have been invalidated. The _M_ptr(__r._M_ptr)
01170       //  conversion may require access to *__r._M_ptr (virtual inheritance).
01171       //
01172       // It is not possible to avoid spurious access violations since
01173       // in multithreaded programs __r._M_ptr may be invalidated at any point.
01174       template<typename _Tp1, typename = typename
01175            std::enable_if<std::is_convertible<_Tp1*, _Tp*>::value>::type>
01176     __weak_ptr(const __weak_ptr<_Tp1, _Lp>& __r)
01177     : _M_refcount(__r._M_refcount) // never throws
01178         { _M_ptr = __r.lock().get(); }
01179 
01180       template<typename _Tp1, typename = typename
01181            std::enable_if<std::is_convertible<_Tp1*, _Tp*>::value>::type>
01182     __weak_ptr(const __shared_ptr<_Tp1, _Lp>& __r)
01183     : _M_ptr(__r._M_ptr), _M_refcount(__r._M_refcount) // never throws
01184     { }
01185 
01186       template<typename _Tp1>
01187     __weak_ptr&
01188     operator=(const __weak_ptr<_Tp1, _Lp>& __r) // never throws
01189     {
01190       _M_ptr = __r.lock().get();
01191       _M_refcount = __r._M_refcount;
01192       return *this;
01193     }
01194 
01195       template<typename _Tp1>
01196     __weak_ptr&
01197     operator=(const __shared_ptr<_Tp1, _Lp>& __r) // never throws
01198     {
01199       _M_ptr = __r._M_ptr;
01200       _M_refcount = __r._M_refcount;
01201       return *this;
01202     }
01203 
01204       __shared_ptr<_Tp, _Lp>
01205       lock() const // never throws
01206       {
01207 #ifdef __GTHREADS
01208     // Optimization: avoid throw overhead.
01209     if (expired())
01210       return __shared_ptr<element_type, _Lp>();
01211 
01212     __try
01213       {
01214         return __shared_ptr<element_type, _Lp>(*this);
01215       }
01216     __catch(const bad_weak_ptr&)
01217       {
01218         // Q: How can we get here?
01219         // A: Another thread may have invalidated r after the
01220         //    use_count test above.
01221         return __shared_ptr<element_type, _Lp>();
01222       }
01223 
01224 #else
01225     // Optimization: avoid try/catch overhead when single threaded.
01226     return expired() ? __shared_ptr<element_type, _Lp>()
01227              : __shared_ptr<element_type, _Lp>(*this);
01228 
01229 #endif
01230       } // XXX MT
01231 
01232       long
01233       use_count() const // never throws
01234       { return _M_refcount._M_get_use_count(); }
01235 
01236       bool
01237       expired() const // never throws
01238       { return _M_refcount._M_get_use_count() == 0; }
01239 
01240       template<typename _Tp1>
01241     bool
01242     owner_before(const __shared_ptr<_Tp1, _Lp>& __rhs) const
01243     { return _M_refcount._M_less(__rhs._M_refcount); }
01244 
01245       template<typename _Tp1>
01246     bool
01247     owner_before(const __weak_ptr<_Tp1, _Lp>& __rhs) const
01248     { return _M_refcount._M_less(__rhs._M_refcount); }
01249 
01250       void
01251       reset() // never throws
01252       { __weak_ptr().swap(*this); }
01253 
01254       void
01255       swap(__weak_ptr& __s) // never throws
01256       {
01257     std::swap(_M_ptr, __s._M_ptr);
01258     _M_refcount._M_swap(__s._M_refcount);
01259       }
01260 
01261     private:
01262       // Used by __enable_shared_from_this.
01263       void
01264       _M_assign(_Tp* __ptr, const __shared_count<_Lp>& __refcount)
01265       {
01266     _M_ptr = __ptr;
01267     _M_refcount = __refcount;
01268       }
01269 
01270       template<typename _Tp1, _Lock_policy _Lp1> friend class __shared_ptr;
01271       template<typename _Tp1, _Lock_policy _Lp1> friend class __weak_ptr;
01272       friend class __enable_shared_from_this<_Tp, _Lp>;
01273       friend class enable_shared_from_this<_Tp>;
01274 
01275       _Tp*       _M_ptr;         // Contained pointer.
01276       __weak_count<_Lp>  _M_refcount;    // Reference counter.
01277     };
01278 
01279   // 20.8.13.3.7 weak_ptr specialized algorithms.
01280   template<typename _Tp, _Lock_policy _Lp>
01281     inline void
01282     swap(__weak_ptr<_Tp, _Lp>& __a, __weak_ptr<_Tp, _Lp>& __b)
01283     { __a.swap(__b); }
01284 
01285   template<typename _Tp, typename _Tp1>
01286     struct _Sp_owner_less : public binary_function<_Tp, _Tp, bool>
01287     {
01288       bool
01289       operator()(const _Tp& __lhs, const _Tp& __rhs) const
01290       { return __lhs.owner_before(__rhs); }
01291 
01292       bool
01293       operator()(const _Tp& __lhs, const _Tp1& __rhs) const
01294       { return __lhs.owner_before(__rhs); }
01295 
01296       bool
01297       operator()(const _Tp1& __lhs, const _Tp& __rhs) const
01298       { return __lhs.owner_before(__rhs); }
01299     };
01300 
01301   template<typename _Tp, _Lock_policy _Lp>
01302     struct owner_less<__shared_ptr<_Tp, _Lp>>
01303     : public _Sp_owner_less<__shared_ptr<_Tp, _Lp>, __weak_ptr<_Tp, _Lp>>
01304     { };
01305 
01306   template<typename _Tp, _Lock_policy _Lp>
01307     struct owner_less<__weak_ptr<_Tp, _Lp>>
01308     : public _Sp_owner_less<__weak_ptr<_Tp, _Lp>, __shared_ptr<_Tp, _Lp>>
01309     { };
01310 
01311 
01312   template<typename _Tp, _Lock_policy _Lp>
01313     class __enable_shared_from_this
01314     {
01315     protected:
01316       constexpr __enable_shared_from_this() { }
01317 
01318       __enable_shared_from_this(const __enable_shared_from_this&) { }
01319 
01320       __enable_shared_from_this&
01321       operator=(const __enable_shared_from_this&)
01322       { return *this; }
01323 
01324       ~__enable_shared_from_this() { }
01325 
01326     public:
01327       __shared_ptr<_Tp, _Lp>
01328       shared_from_this()
01329       { return __shared_ptr<_Tp, _Lp>(this->_M_weak_this); }
01330 
01331       __shared_ptr<const _Tp, _Lp>
01332       shared_from_this() const
01333       { return __shared_ptr<const _Tp, _Lp>(this->_M_weak_this); }
01334 
01335     private:
01336       template<typename _Tp1>
01337     void
01338     _M_weak_assign(_Tp1* __p, const __shared_count<_Lp>& __n) const
01339     { _M_weak_this._M_assign(__p, __n); }
01340 
01341       template<typename _Tp1>
01342     friend void
01343     __enable_shared_from_this_helper(const __shared_count<_Lp>& __pn,
01344                      const __enable_shared_from_this* __pe,
01345                      const _Tp1* __px)
01346     {
01347       if (__pe != 0)
01348         __pe->_M_weak_assign(const_cast<_Tp1*>(__px), __pn);
01349     }
01350 
01351       mutable __weak_ptr<_Tp, _Lp>  _M_weak_this;
01352     };
01353 
01354 
01355   template<typename _Tp, _Lock_policy _Lp, typename _Alloc, typename... _Args>
01356     inline __shared_ptr<_Tp, _Lp>
01357     __allocate_shared(const _Alloc& __a, _Args&&... __args)
01358     {
01359       return __shared_ptr<_Tp, _Lp>(_Sp_make_shared_tag(), __a,
01360                     std::forward<_Args>(__args)...);
01361     }
01362 
01363   template<typename _Tp, _Lock_policy _Lp, typename... _Args>
01364     inline __shared_ptr<_Tp, _Lp>
01365     __make_shared(_Args&&... __args)
01366     {
01367       typedef typename std::remove_const<_Tp>::type _Tp_nc;
01368       return __allocate_shared<_Tp, _Lp>(std::allocator<_Tp_nc>(),
01369                      std::forward<_Args>(__args)...);
01370     }
01371 
01372   /// std::hash specialization for __shared_ptr.
01373   template<typename _Tp, _Lock_policy _Lp>
01374     struct hash<__shared_ptr<_Tp, _Lp>>
01375     : public std::unary_function<__shared_ptr<_Tp, _Lp>, size_t>
01376     {
01377       size_t
01378       operator()(const __shared_ptr<_Tp, _Lp>& __s) const
01379       { return std::hash<_Tp*>()(__s.get()); }
01380     };
01381 
01382 _GLIBCXX_END_NAMESPACE_VERSION
01383 } // namespace
01384 
01385 #endif // _SHARED_PTR_BASE_H
Generated by  doxygen 1.7.1