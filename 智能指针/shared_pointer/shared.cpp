include
bits
shared_ptr.h
Go to the documentation of this file.
00001 // shared_ptr and weak_ptr implementation -*- C++ -*-
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
00044 /** @file bits/shared_ptr.h
00045  *  This is an internal header file, included by other library headers.
00046  *  Do not attempt to use it directly. @headername{memory}
00047  */
00048 
00049 #ifndef _SHARED_PTR_H
00050 #define _SHARED_PTR_H 1
00051 
00052 #include <bits/shared_ptr_base.h>
00053 
00054 namespace std _GLIBCXX_VISIBILITY(default)
00055 {
00056 _GLIBCXX_BEGIN_NAMESPACE_VERSION
00057 
00058   /**
00059    * @addtogroup pointer_abstractions
00060    * @{
00061    */
00062 
00063   /// 2.2.3.7 shared_ptr I/O
00064   template<typename _Ch, typename _Tr, typename _Tp, _Lock_policy _Lp>
00065     inline std::basic_ostream<_Ch, _Tr>&
00066     operator<<(std::basic_ostream<_Ch, _Tr>& __os,
00067            const __shared_ptr<_Tp, _Lp>& __p)
00068     {
00069       __os << __p.get();
00070       return __os;
00071     }
00072 
00073   /// 2.2.3.10 shared_ptr get_deleter (experimental)
00074   template<typename _Del, typename _Tp, _Lock_policy _Lp>
00075     inline _Del*
00076     get_deleter(const __shared_ptr<_Tp, _Lp>& __p)
00077     {
00078 #ifdef __GXX_RTTI
00079       return static_cast<_Del*>(__p._M_get_deleter(typeid(_Del)));
00080 #else
00081       return 0;
00082 #endif
00083     }
00084 
00085 
00086   /**
00087    *  @brief  A smart pointer with reference-counted copy semantics.
00088    *
00089    *  The object pointed to is deleted when the last shared_ptr pointing to
00090    *  it is destroyed or reset.
00091   */
00092   template<typename _Tp>
00093     class shared_ptr : public __shared_ptr<_Tp>
00094     {
00095     public:
00096       /**
00097        *  @brief  Construct an empty %shared_ptr.
00098        *  @post   use_count()==0 && get()==0
00099        */
00100       constexpr shared_ptr()
00101       : __shared_ptr<_Tp>() { }
00102 
00103       /**
00104        *  @brief  Construct a %shared_ptr that owns the pointer @a __p.
00105        *  @param  __p  A pointer that is convertible to element_type*.
00106        *  @post   use_count() == 1 && get() == __p
00107        *  @throw  std::bad_alloc, in which case @c delete @a __p is called.
00108        */
00109       template<typename _Tp1>
00110     explicit shared_ptr(_Tp1* __p)
00111         : __shared_ptr<_Tp>(__p) { }
00112 
00113       /**
00114        *  @brief  Construct a %shared_ptr that owns the pointer @a __p
00115        *          and the deleter @a __d.
00116        *  @param  __p  A pointer.
00117        *  @param  __d  A deleter.
00118        *  @post   use_count() == 1 && get() == __p
00119        *  @throw  std::bad_alloc, in which case @a __d(__p) is called.
00120        *
00121        *  Requirements: _Deleter's copy constructor and destructor must
00122        *  not throw
00123        *
00124        *  __shared_ptr will release __p by calling __d(__p)
00125        */
00126       template<typename _Tp1, typename _Deleter>
00127     shared_ptr(_Tp1* __p, _Deleter __d)
00128         : __shared_ptr<_Tp>(__p, __d) { }
00129 
00130       /**
00131        *  @brief  Construct a %shared_ptr that owns a null pointer
00132        *          and the deleter @a __d.
00133        *  @param  __p  A null pointer constant.
00134        *  @param  __d  A deleter.
00135        *  @post   use_count() == 1 && get() == __p
00136        *  @throw  std::bad_alloc, in which case @a __d(__p) is called.
00137        *
00138        *  Requirements: _Deleter's copy constructor and destructor must
00139        *  not throw
00140        *
00141        *  The last owner will call __d(__p)
00142        */
00143       template<typename _Deleter>
00144     shared_ptr(nullptr_t __p, _Deleter __d)
00145         : __shared_ptr<_Tp>(__p, __d) { }
00146 
00147       /**
00148        *  @brief  Construct a %shared_ptr that owns the pointer @a __p
00149        *          and the deleter @a __d.
00150        *  @param  __p  A pointer.
00151        *  @param  __d  A deleter.
00152        *  @param  __a  An allocator.
00153        *  @post   use_count() == 1 && get() == __p
00154        *  @throw  std::bad_alloc, in which case @a __d(__p) is called.
00155        *
00156        *  Requirements: _Deleter's copy constructor and destructor must
00157        *  not throw _Alloc's copy constructor and destructor must not
00158        *  throw.
00159        *
00160        *  __shared_ptr will release __p by calling __d(__p)
00161        */
00162       template<typename _Tp1, typename _Deleter, typename _Alloc>
00163     shared_ptr(_Tp1* __p, _Deleter __d, _Alloc __a)
00164     : __shared_ptr<_Tp>(__p, __d, std::move(__a)) { }
00165 
00166       /**
00167        *  @brief  Construct a %shared_ptr that owns a null pointer
00168        *          and the deleter @a __d.
00169        *  @param  __p  A null pointer constant.
00170        *  @param  __d  A deleter.
00171        *  @param  __a  An allocator.
00172        *  @post   use_count() == 1 && get() == __p
00173        *  @throw  std::bad_alloc, in which case @a __d(__p) is called.
00174        *
00175        *  Requirements: _Deleter's copy constructor and destructor must
00176        *  not throw _Alloc's copy constructor and destructor must not
00177        *  throw.
00178        *
00179        *  The last owner will call __d(__p)
00180        */
00181       template<typename _Deleter, typename _Alloc>
00182     shared_ptr(nullptr_t __p, _Deleter __d, _Alloc __a)
00183     : __shared_ptr<_Tp>(__p, __d, std::move(__a)) { }
00184 
00185       // Aliasing constructor
00186 
00187       /**
00188        *  @brief  Constructs a %shared_ptr instance that stores @a __p
00189        *          and shares ownership with @a __r.
00190        *  @param  __r  A %shared_ptr.
00191        *  @param  __p  A pointer that will remain valid while @a *__r is valid.
00192        *  @post   get() == __p && use_count() == __r.use_count()
00193        *
00194        *  This can be used to construct a @c shared_ptr to a sub-object
00195        *  of an object managed by an existing @c shared_ptr.
00196        *
00197        * @code
00198        * shared_ptr< pair<int,int> > pii(new pair<int,int>());
00199        * shared_ptr<int> pi(pii, &pii->first);
00200        * assert(pii.use_count() == 2);
00201        * @endcode
00202        */
00203       template<typename _Tp1>
00204     shared_ptr(const shared_ptr<_Tp1>& __r, _Tp* __p)
00205     : __shared_ptr<_Tp>(__r, __p) { }
00206 
00207       /**
00208        *  @brief  If @a __r is empty, constructs an empty %shared_ptr;
00209        *          otherwise construct a %shared_ptr that shares ownership
00210        *          with @a __r.
00211        *  @param  __r  A %shared_ptr.
00212        *  @post   get() == __r.get() && use_count() == __r.use_count()
00213        */
00214       template<typename _Tp1, typename = typename
00215            std::enable_if<std::is_convertible<_Tp1*, _Tp*>::value>::type>
00216     shared_ptr(const shared_ptr<_Tp1>& __r)
00217         : __shared_ptr<_Tp>(__r) { }
00218 
00219       /**
00220        *  @brief  Move-constructs a %shared_ptr instance from @a __r.
00221        *  @param  __r  A %shared_ptr rvalue.
00222        *  @post   *this contains the old value of @a __r, @a __r is empty.
00223        */
00224       shared_ptr(shared_ptr&& __r)
00225       : __shared_ptr<_Tp>(std::move(__r)) { }
00226 
00227       /**
00228        *  @brief  Move-constructs a %shared_ptr instance from @a __r.
00229        *  @param  __r  A %shared_ptr rvalue.
00230        *  @post   *this contains the old value of @a __r, @a __r is empty.
00231        */
00232       template<typename _Tp1, typename = typename
00233            std::enable_if<std::is_convertible<_Tp1*, _Tp*>::value>::type>
00234     shared_ptr(shared_ptr<_Tp1>&& __r)
00235     : __shared_ptr<_Tp>(std::move(__r)) { }
00236 
00237       /**
00238        *  @brief  Constructs a %shared_ptr that shares ownership with @a __r
00239        *          and stores a copy of the pointer stored in @a __r.
00240        *  @param  __r  A weak_ptr.
00241        *  @post   use_count() == __r.use_count()
00242        *  @throw  bad_weak_ptr when __r.expired(),
00243        *          in which case the constructor has no effect.
00244        */
00245       template<typename _Tp1>
00246     explicit shared_ptr(const weak_ptr<_Tp1>& __r)
00247     : __shared_ptr<_Tp>(__r) { }
00248 
00249 #if _GLIBCXX_USE_DEPRECATED
00250       template<typename _Tp1>
00251     shared_ptr(std::auto_ptr<_Tp1>&& __r)
00252     : __shared_ptr<_Tp>(std::move(__r)) { }
00253 #endif
00254 
00255       template<typename _Tp1, typename _Del>
00256     shared_ptr(std::unique_ptr<_Tp1, _Del>&& __r)
00257     : __shared_ptr<_Tp>(std::move(__r)) { }
00258 
00259       /**
00260        *  @brief  Construct an empty %shared_ptr.
00261        *  @param  __p  A null pointer constant.
00262        *  @post   use_count() == 0 && get() == nullptr
00263        */
00264       constexpr shared_ptr(nullptr_t __p)
00265       : __shared_ptr<_Tp>(__p) { }
00266 
00267       template<typename _Tp1>
00268     shared_ptr&
00269     operator=(const shared_ptr<_Tp1>& __r) // never throws
00270     {
00271       this->__shared_ptr<_Tp>::operator=(__r);
00272       return *this;
00273     }
00274 
00275 #if _GLIBCXX_USE_DEPRECATED
00276       template<typename _Tp1>
00277     shared_ptr&
00278     operator=(std::auto_ptr<_Tp1>&& __r)
00279     {
00280       this->__shared_ptr<_Tp>::operator=(std::move(__r));
00281       return *this;
00282     }
00283 #endif
00284 
00285       shared_ptr&
00286       operator=(shared_ptr&& __r)
00287       {
00288     this->__shared_ptr<_Tp>::operator=(std::move(__r));
00289     return *this;
00290       }
00291 
00292       template<class _Tp1>
00293     shared_ptr&
00294     operator=(shared_ptr<_Tp1>&& __r)
00295     {
00296       this->__shared_ptr<_Tp>::operator=(std::move(__r));
00297       return *this;
00298     }
00299 
00300       template<typename _Tp1, typename _Del>
00301     shared_ptr&
00302     operator=(std::unique_ptr<_Tp1, _Del>&& __r)
00303     {
00304       this->__shared_ptr<_Tp>::operator=(std::move(__r));
00305       return *this;
00306     }
00307 
00308     private:
00309       // This constructor is non-standard, it is used by allocate_shared.
00310       template<typename _Alloc, typename... _Args>
00311     shared_ptr(_Sp_make_shared_tag __tag, const _Alloc& __a,
00312            _Args&&... __args)
00313     : __shared_ptr<_Tp>(__tag, __a, std::forward<_Args>(__args)...)
00314     { }
00315 
00316       template<typename _Tp1, typename _Alloc, typename... _Args>
00317     friend shared_ptr<_Tp1>
00318     allocate_shared(const _Alloc& __a, _Args&&... __args);
00319     };
00320 
00321   // 20.8.13.2.7 shared_ptr comparisons
00322   template<typename _Tp1, typename _Tp2>
00323     inline bool
00324     operator==(const shared_ptr<_Tp1>& __a, const shared_ptr<_Tp2>& __b)
00325     { return __a.get() == __b.get(); }
00326 
00327   template<typename _Tp>
00328     inline bool
00329     operator==(const shared_ptr<_Tp>& __a, nullptr_t)
00330     { return __a.get() == nullptr; }
00331 
00332   template<typename _Tp>
00333     inline bool
00334     operator==(nullptr_t, const shared_ptr<_Tp>& __b)
00335     { return nullptr == __b.get(); }
00336 
00337   template<typename _Tp1, typename _Tp2>
00338     inline bool
00339     operator!=(const shared_ptr<_Tp1>& __a, const shared_ptr<_Tp2>& __b)
00340     { return __a.get() != __b.get(); }
00341 
00342   template<typename _Tp>
00343     inline bool
00344     operator!=(const shared_ptr<_Tp>& __a, nullptr_t)
00345     { return __a.get() != nullptr; }
00346 
00347   template<typename _Tp>
00348     inline bool
00349     operator!=(nullptr_t, const shared_ptr<_Tp>& __b)
00350     { return nullptr != __b.get(); }
00351 
00352   template<typename _Tp1, typename _Tp2>
00353     inline bool
00354     operator<(const shared_ptr<_Tp1>& __a, const shared_ptr<_Tp2>& __b)
00355     { return __a.get() < __b.get(); }
00356 
00357   template<typename _Tp>
00358     struct less<shared_ptr<_Tp>> : public _Sp_less<shared_ptr<_Tp>>
00359     { };
00360 
00361   // 20.8.13.2.9 shared_ptr specialized algorithms.
00362   template<typename _Tp>
00363     inline void
00364     swap(shared_ptr<_Tp>& __a, shared_ptr<_Tp>& __b)
00365     { __a.swap(__b); }
00366 
00367   // 20.8.13.2.10 shared_ptr casts.
00368   template<typename _Tp, typename _Tp1>
00369     inline shared_ptr<_Tp>
00370     static_pointer_cast(const shared_ptr<_Tp1>& __r)
00371     { return shared_ptr<_Tp>(__r, static_cast<_Tp*>(__r.get())); }
00372 
00373   template<typename _Tp, typename _Tp1>
00374     inline shared_ptr<_Tp>
00375     const_pointer_cast(const shared_ptr<_Tp1>& __r)
00376     { return shared_ptr<_Tp>(__r, const_cast<_Tp*>(__r.get())); }
00377 
00378   template<typename _Tp, typename _Tp1>
00379     inline shared_ptr<_Tp>
00380     dynamic_pointer_cast(const shared_ptr<_Tp1>& __r)
00381     {
00382       if (_Tp* __p = dynamic_cast<_Tp*>(__r.get()))
00383     return shared_ptr<_Tp>(__r, __p);
00384       return shared_ptr<_Tp>();
00385     }
00386 
00387 
00388   /**
00389    *  @brief  A smart pointer with weak semantics.
00390    *
00391    *  With forwarding constructors and assignment operators.
00392    */
00393   template<typename _Tp>
00394     class weak_ptr : public __weak_ptr<_Tp>
00395     {
00396     public:
00397       constexpr weak_ptr()
00398       : __weak_ptr<_Tp>() { }
00399 
00400       template<typename _Tp1, typename = typename
00401            std::enable_if<std::is_convertible<_Tp1*, _Tp*>::value>::type>
00402     weak_ptr(const weak_ptr<_Tp1>& __r)
00403     : __weak_ptr<_Tp>(__r) { }
00404 
00405       template<typename _Tp1, typename = typename
00406            std::enable_if<std::is_convertible<_Tp1*, _Tp*>::value>::type>
00407     weak_ptr(const shared_ptr<_Tp1>& __r)
00408     : __weak_ptr<_Tp>(__r) { }
00409 
00410       template<typename _Tp1>
00411     weak_ptr&
00412     operator=(const weak_ptr<_Tp1>& __r) // never throws
00413     {
00414       this->__weak_ptr<_Tp>::operator=(__r);
00415       return *this;
00416     }
00417 
00418       template<typename _Tp1>
00419     weak_ptr&
00420     operator=(const shared_ptr<_Tp1>& __r) // never throws
00421     {
00422       this->__weak_ptr<_Tp>::operator=(__r);
00423       return *this;
00424     }
00425 
00426       shared_ptr<_Tp>
00427       lock() const // never throws
00428       {
00429 #ifdef __GTHREADS
00430     if (this->expired())
00431       return shared_ptr<_Tp>();
00432 
00433     __try
00434       {
00435         return shared_ptr<_Tp>(*this);
00436       }
00437     __catch(const bad_weak_ptr&)
00438       {
00439         return shared_ptr<_Tp>();
00440       }
00441 #else
00442     return this->expired() ? shared_ptr<_Tp>() : shared_ptr<_Tp>(*this);
00443 #endif
00444       }
00445     };
00446 
00447   // 20.8.13.3.7 weak_ptr specialized algorithms.
00448   template<typename _Tp>
00449     inline void
00450     swap(weak_ptr<_Tp>& __a, weak_ptr<_Tp>& __b)
00451     { __a.swap(__b); }
00452 
00453 
00454   /// Primary template owner_less
00455   template<typename _Tp>
00456     struct owner_less;
00457 
00458   /// Partial specialization of owner_less for shared_ptr.
00459   template<typename _Tp>
00460     struct owner_less<shared_ptr<_Tp>>
00461     : public _Sp_owner_less<shared_ptr<_Tp>, weak_ptr<_Tp>>
00462     { };
00463 
00464   /// Partial specialization of owner_less for weak_ptr.
00465   template<typename _Tp>
00466     struct owner_less<weak_ptr<_Tp>>
00467     : public _Sp_owner_less<weak_ptr<_Tp>, shared_ptr<_Tp>>
00468     { };
00469 
00470   /**
00471    *  @brief Base class allowing use of member function shared_from_this.
00472    */
00473   template<typename _Tp>
00474     class enable_shared_from_this
00475     {
00476     protected:
00477       constexpr enable_shared_from_this() { }
00478 
00479       enable_shared_from_this(const enable_shared_from_this&) { }
00480 
00481       enable_shared_from_this&
00482       operator=(const enable_shared_from_this&)
00483       { return *this; }
00484 
00485       ~enable_shared_from_this() { }
00486 
00487     public:
00488       shared_ptr<_Tp>
00489       shared_from_this()
00490       { return shared_ptr<_Tp>(this->_M_weak_this); }
00491 
00492       shared_ptr<const _Tp>
00493       shared_from_this() const
00494       { return shared_ptr<const _Tp>(this->_M_weak_this); }
00495 
00496     private:
00497       template<typename _Tp1>
00498     void
00499     _M_weak_assign(_Tp1* __p, const __shared_count<>& __n) const
00500     { _M_weak_this._M_assign(__p, __n); }
00501 
00502       template<typename _Tp1>
00503     friend void
00504     __enable_shared_from_this_helper(const __shared_count<>& __pn,
00505                      const enable_shared_from_this* __pe,
00506                      const _Tp1* __px)
00507     {
00508       if (__pe != 0)
00509         __pe->_M_weak_assign(const_cast<_Tp1*>(__px), __pn);
00510     }
00511 
00512       mutable weak_ptr<_Tp>  _M_weak_this;
00513     };
00514 
00515   /**
00516    *  @brief  Create an object that is owned by a shared_ptr.
00517    *  @param  __a     An allocator.
00518    *  @param  __args  Arguments for the @a _Tp object's constructor.
00519    *  @return A shared_ptr that owns the newly created object.
00520    *  @throw  An exception thrown from @a _Alloc::allocate or from the
00521    *          constructor of @a _Tp.
00522    *
00523    *  A copy of @a __a will be used to allocate memory for the shared_ptr
00524    *  and the new object.
00525    */
00526   template<typename _Tp, typename _Alloc, typename... _Args>
00527     inline shared_ptr<_Tp>
00528     allocate_shared(const _Alloc& __a, _Args&&... __args)
00529     {
00530       return shared_ptr<_Tp>(_Sp_make_shared_tag(), __a,
00531                  std::forward<_Args>(__args)...);
00532     }
00533 
00534   /**
00535    *  @brief  Create an object that is owned by a shared_ptr.
00536    *  @param  __args  Arguments for the @a _Tp object's constructor.
00537    *  @return A shared_ptr that owns the newly created object.
00538    *  @throw  std::bad_alloc, or an exception thrown from the
00539    *          constructor of @a _Tp.
00540    */
00541   template<typename _Tp, typename... _Args>
00542     inline shared_ptr<_Tp>
00543     make_shared(_Args&&... __args)
00544     {
00545       typedef typename std::remove_const<_Tp>::type _Tp_nc;
00546       return allocate_shared<_Tp>(std::allocator<_Tp_nc>(),
00547                   std::forward<_Args>(__args)...);
00548     }
00549 
00550   /// std::hash specialization for shared_ptr.
00551   template<typename _Tp>
00552     struct hash<shared_ptr<_Tp>>
00553     : public std::unary_function<shared_ptr<_Tp>, size_t>
00554     {
00555       size_t
00556       operator()(const shared_ptr<_Tp>& __s) const
00557       { return std::hash<_Tp*>()(__s.get()); }
00558     };
00559 
00560   // @} group pointer_abstractions
00561 
00562 _GLIBCXX_END_NAMESPACE_VERSION
00563 } // namespace
00564 
00565 #endif // _SHARED_PTR_H
Generated by  doxygen 1.7.1