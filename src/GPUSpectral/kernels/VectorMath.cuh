#pragma once

#include <vector_functions.h>
#include <vector_types.h>

#if !defined(__CUDACC_RTC__)
#include <cmath>
#include <cstdlib>
#endif

/* scalar functions used in vector functions */
#ifndef M_PIf
#define M_PIf       3.14159265358979323846f
#endif
#ifndef M_PI_2f
#define M_PI_2f     1.57079632679489661923f
#endif
#ifndef M_1_PIf
#define M_1_PIf     0.318309886183790671538f
#endif


#if defined(__CUDACC__) || defined(__CUDABE__)
#    define HOSTDEVICE __host__ __device__
#    define CUDAINLINE __forceinline__
#    define CONST_STATIC_INIT( ... )
#else
#define HOSTDEVICE 
#define CUDAINLINE inline 
#define CONST_STATIC_INIT( ... ) = __VA_ARGS__
#endif

#if !defined(__CUDACC__)

CUDAINLINE HOSTDEVICE int max(int a, int b)
{
    return a > b ? a : b;
}

CUDAINLINE HOSTDEVICE int min(int a, int b)
{
    return a < b ? a : b;
}

CUDAINLINE HOSTDEVICE long long max(long long a, long long b)
{
    return a > b ? a : b;
}

CUDAINLINE HOSTDEVICE long long min(long long a, long long b)
{
    return a < b ? a : b;
}

CUDAINLINE HOSTDEVICE unsigned int max(unsigned int a, unsigned int b)
{
    return a > b ? a : b;
}

CUDAINLINE HOSTDEVICE unsigned int min(unsigned int a, unsigned int b)
{
    return a < b ? a : b;
}

CUDAINLINE HOSTDEVICE unsigned long long max(unsigned long long a, unsigned long long b)
{
    return a > b ? a : b;
}

CUDAINLINE HOSTDEVICE unsigned long long min(unsigned long long a, unsigned long long b)
{
    return a < b ? a : b;
}


/** lerp */
CUDAINLINE HOSTDEVICE float lerp(const float a, const float b, const float t)
{
  return a + t*(b-a);
}

/** bilerp */
CUDAINLINE HOSTDEVICE float bilerp(const float x00, const float x10, const float x01, const float x11,
                                         const float u, const float v)
{
  return lerp( lerp( x00, x10, u ), lerp( x01, x11, u ), v );
}

template <typename IntegerType>
CUDAINLINE HOSTDEVICE IntegerType roundUp(IntegerType x, IntegerType y)
{
    return ( ( x + y - 1 ) / y ) * y;
}

#endif

/** clamp */
CUDAINLINE HOSTDEVICE float clamp( const float f, const float a, const float b )
{
    return fmaxf( a, fminf( f, b ) );
}


/* float2 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/
CUDAINLINE HOSTDEVICE float2 make_float2(const float s)
{
  return make_float2(s, s);
}
CUDAINLINE HOSTDEVICE float2 make_float2(const int2& a)
{
  return make_float2(float(a.x), float(a.y));
}
CUDAINLINE HOSTDEVICE float2 make_float2(const uint2& a)
{
  return make_float2(float(a.x), float(a.y));
}
/** @} */

/** negate */
CUDAINLINE HOSTDEVICE float2 operator-(const float2& a)
{
  return make_float2(-a.x, -a.y);
}

/** min 
* @{
*/
CUDAINLINE HOSTDEVICE float2 fminf(const float2& a, const float2& b)
{
  return make_float2(fminf(a.x,b.x), fminf(a.y,b.y));
}
CUDAINLINE HOSTDEVICE float fminf(const float2& a)
{
  return fminf(a.x, a.y);
}
/** @} */

/** max 
* @{
*/
CUDAINLINE HOSTDEVICE float2 fmaxf(const float2& a, const float2& b)
{
  return make_float2(fmaxf(a.x,b.x), fmaxf(a.y,b.y));
}
CUDAINLINE HOSTDEVICE float fmaxf(const float2& a)
{
  return fmaxf(a.x, a.y);
}
/** @} */

/** add 
* @{
*/
CUDAINLINE HOSTDEVICE float2 operator+(const float2& a, const float2& b)
{
  return make_float2(a.x + b.x, a.y + b.y);
}
CUDAINLINE HOSTDEVICE float2 operator+(const float2& a, const float b)
{
  return make_float2(a.x + b, a.y + b);
}
CUDAINLINE HOSTDEVICE float2 operator+(const float a, const float2& b)
{
  return make_float2(a + b.x, a + b.y);
}
CUDAINLINE HOSTDEVICE void operator+=(float2& a, const float2& b)
{
  a.x += b.x; a.y += b.y;
}
/** @} */

/** subtract 
* @{
*/
CUDAINLINE HOSTDEVICE float2 operator-(const float2& a, const float2& b)
{
  return make_float2(a.x - b.x, a.y - b.y);
}
CUDAINLINE HOSTDEVICE float2 operator-(const float2& a, const float b)
{
  return make_float2(a.x - b, a.y - b);
}
CUDAINLINE HOSTDEVICE float2 operator-(const float a, const float2& b)
{
  return make_float2(a - b.x, a - b.y);
}
CUDAINLINE HOSTDEVICE void operator-=(float2& a, const float2& b)
{
  a.x -= b.x; a.y -= b.y;
}
/** @} */

/** multiply 
* @{
*/
CUDAINLINE HOSTDEVICE float2 operator*(const float2& a, const float2& b)
{
  return make_float2(a.x * b.x, a.y * b.y);
}
CUDAINLINE HOSTDEVICE float2 operator*(const float2& a, const float s)
{
  return make_float2(a.x * s, a.y * s);
}
CUDAINLINE HOSTDEVICE float2 operator*(const float s, const float2& a)
{
  return make_float2(a.x * s, a.y * s);
}
CUDAINLINE HOSTDEVICE void operator*=(float2& a, const float2& s)
{
  a.x *= s.x; a.y *= s.y;
}
CUDAINLINE HOSTDEVICE void operator*=(float2& a, const float s)
{
  a.x *= s; a.y *= s;
}
/** @} */

/** divide 
* @{
*/
CUDAINLINE HOSTDEVICE float2 operator/(const float2& a, const float2& b)
{
  return make_float2(a.x / b.x, a.y / b.y);
}
CUDAINLINE HOSTDEVICE float2 operator/(const float2& a, const float s)
{
  float inv = 1.0f / s;
  return a * inv;
}
CUDAINLINE HOSTDEVICE float2 operator/(const float s, const float2& a)
{
  return make_float2( s/a.x, s/a.y );
}
CUDAINLINE HOSTDEVICE void operator/=(float2& a, const float s)
{
  float inv = 1.0f / s;
  a *= inv;
}
/** @} */

/** lerp */
CUDAINLINE HOSTDEVICE float2 lerp(const float2& a, const float2& b, const float t)
{
  return a + t*(b-a);
}

/** bilerp */
CUDAINLINE HOSTDEVICE float2 bilerp(const float2& x00, const float2& x10, const float2& x01, const float2& x11,
                                          const float u, const float v)
{
  return lerp( lerp( x00, x10, u ), lerp( x01, x11, u ), v );
}

/** clamp 
* @{
*/
CUDAINLINE HOSTDEVICE float2 clamp(const float2& v, const float a, const float b)
{
  return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}

CUDAINLINE HOSTDEVICE float2 clamp(const float2& v, const float2& a, const float2& b)
{
  return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
/** @} */

/** dot product */
CUDAINLINE HOSTDEVICE float dot(const float2& a, const float2& b)
{
  return a.x * b.x + a.y * b.y;
}

/** length */
CUDAINLINE HOSTDEVICE float length(const float2& v)
{
  return sqrtf(dot(v, v));
}

/** normalize */
CUDAINLINE HOSTDEVICE float2 normalize(const float2& v)
{
  float invLen = 1.0f / sqrtf(dot(v, v));
  return v * invLen;
}

/** floor */
CUDAINLINE HOSTDEVICE float2 floor(const float2& v)
{
  return make_float2(::floorf(v.x), ::floorf(v.y));
}

/** reflect */
CUDAINLINE HOSTDEVICE float2 reflect(const float2& i, const float2& n)
{
  return i - 2.0f * n * dot(n,i);
}

/** Faceforward
* Returns N if dot(i, nref) > 0; else -N; 
* Typical usage is N = faceforward(N, -ray.dir, N);
* Note that this is opposite of what faceforward does in Cg and GLSL */
CUDAINLINE HOSTDEVICE float2 faceforward(const float2& n, const float2& i, const float2& nref)
{
  return n * copysignf( 1.0f, dot(i, nref) );
}

/** exp */
CUDAINLINE HOSTDEVICE float2 expf(const float2& v)
{
  return make_float2(::expf(v.x), ::expf(v.y));
}

/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE float getByIndex(const float2& v, int i)
{
  return ((float*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE void setByIndex(float2& v, int i, float x)
{
  ((float*)(&v))[i] = x;
}


/* float3 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/
CUDAINLINE HOSTDEVICE float3 make_float3(const float s)
{
  return make_float3(s, s, s);
}
CUDAINLINE HOSTDEVICE float3 make_float3(const float2& a)
{
  return make_float3(a.x, a.y, 0.0f);
}
CUDAINLINE HOSTDEVICE float3 make_float3(const int3& a)
{
  return make_float3(float(a.x), float(a.y), float(a.z));
}
CUDAINLINE HOSTDEVICE float3 make_float3(const uint3& a)
{
  return make_float3(float(a.x), float(a.y), float(a.z));
}
/** @} */

/** negate */
CUDAINLINE HOSTDEVICE float3 operator-(const float3& a)
{
  return make_float3(-a.x, -a.y, -a.z);
}

/** min 
* @{
*/
CUDAINLINE HOSTDEVICE float3 fminf(const float3& a, const float3& b)
{
  return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}
CUDAINLINE HOSTDEVICE float fminf(const float3& a)
{
  return fminf(fminf(a.x, a.y), a.z);
}
/** @} */

/** max 
* @{
*/
CUDAINLINE HOSTDEVICE float3 fmaxf(const float3& a, const float3& b)
{
  return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}
CUDAINLINE HOSTDEVICE float fmaxf(const float3& a)
{
  return fmaxf(fmaxf(a.x, a.y), a.z);
}
/** @} */

/** add 
* @{
*/
CUDAINLINE HOSTDEVICE float3 operator+(const float3& a, const float3& b)
{
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
CUDAINLINE HOSTDEVICE float3 operator+(const float3& a, const float b)
{
  return make_float3(a.x + b, a.y + b, a.z + b);
}
CUDAINLINE HOSTDEVICE float3 operator+(const float a, const float3& b)
{
  return make_float3(a + b.x, a + b.y, a + b.z);
}
CUDAINLINE HOSTDEVICE void operator+=(float3& a, const float3& b)
{
  a.x += b.x; a.y += b.y; a.z += b.z;
}
/** @} */

/** subtract 
* @{
*/
CUDAINLINE HOSTDEVICE float3 operator-(const float3& a, const float3& b)
{
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
CUDAINLINE HOSTDEVICE float3 operator-(const float3& a, const float b)
{
  return make_float3(a.x - b, a.y - b, a.z - b);
}
CUDAINLINE HOSTDEVICE float3 operator-(const float a, const float3& b)
{
  return make_float3(a - b.x, a - b.y, a - b.z);
}
CUDAINLINE HOSTDEVICE void operator-=(float3& a, const float3& b)
{
  a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
/** @} */

/** multiply 
* @{
*/
CUDAINLINE HOSTDEVICE float3 operator*(const float3& a, const float3& b)
{
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
CUDAINLINE HOSTDEVICE float3 operator*(const float3& a, const float s)
{
  return make_float3(a.x * s, a.y * s, a.z * s);
}
CUDAINLINE HOSTDEVICE float3 operator*(const float s, const float3& a)
{
  return make_float3(a.x * s, a.y * s, a.z * s);
}
CUDAINLINE HOSTDEVICE void operator*=(float3& a, const float3& s)
{
  a.x *= s.x; a.y *= s.y; a.z *= s.z;
}
CUDAINLINE HOSTDEVICE void operator*=(float3& a, const float s)
{
  a.x *= s; a.y *= s; a.z *= s;
}
/** @} */

/** divide 
* @{
*/
CUDAINLINE HOSTDEVICE float3 operator/(const float3& a, const float3& b)
{
  return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
CUDAINLINE HOSTDEVICE float3 operator/(const float3& a, const float s)
{
  float inv = 1.0f / s;
  return a * inv;
}
CUDAINLINE HOSTDEVICE float3 operator/(const float s, const float3& a)
{
  return make_float3( s/a.x, s/a.y, s/a.z );
}
CUDAINLINE HOSTDEVICE void operator/=(float3& a, const float s)
{
  float inv = 1.0f / s;
  a *= inv;
}
/** @} */

CUDAINLINE HOSTDEVICE float3 pow(const float3& a, const float3& b) {
    return make_float3(pow(a.x, b.x), pow(a.y, b.y), pow(a.z, b.z));
}

/** lerp */
CUDAINLINE HOSTDEVICE float3 lerp(const float3& a, const float3& b, const float t)
{
  return a + t*(b-a);
}

/** bilerp */
CUDAINLINE HOSTDEVICE float3 bilerp(const float3& x00, const float3& x10, const float3& x01, const float3& x11,
                                          const float u, const float v)
{
  return lerp( lerp( x00, x10, u ), lerp( x01, x11, u ), v );
}

/** clamp 
* @{
*/
CUDAINLINE HOSTDEVICE float3 clamp(const float3& v, const float a, const float b)
{
  return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

CUDAINLINE HOSTDEVICE float3 clamp(const float3& v, const float3& a, const float3& b)
{
  return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
/** @} */

/** dot product */
CUDAINLINE HOSTDEVICE float dot(const float3& a, const float3& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

/** cross product */
CUDAINLINE HOSTDEVICE float3 cross(const float3& a, const float3& b)
{
  return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

/** length */
CUDAINLINE HOSTDEVICE float length(const float3& v)
{
  return sqrtf(dot(v, v));
}

/** normalize */
CUDAINLINE HOSTDEVICE float3 normalize(const float3& v)
{
  float invLen = 1.0f / sqrtf(dot(v, v));
  return v * invLen;
}

/** floor */
CUDAINLINE HOSTDEVICE float3 floor(const float3& v)
{
  return make_float3(::floorf(v.x), ::floorf(v.y), ::floorf(v.z));
}

/** reflect */
CUDAINLINE HOSTDEVICE float3 reflect(const float3& i, const float3& n)
{
  return i - 2.0f * n * dot(n,i);
}

/** Faceforward
* Returns N if dot(i, nref) > 0; else -N;
* Typical usage is N = faceforward(N, -ray.dir, N);
* Note that this is opposite of what faceforward does in Cg and GLSL */
CUDAINLINE HOSTDEVICE float3 faceforward(const float3& n, const float3& i, const float3& nref)
{
  return n * copysignf( 1.0f, dot(i, nref) );
}

/** exp */
CUDAINLINE HOSTDEVICE float3 expf(const float3& v)
{
  return make_float3(::expf(v.x), ::expf(v.y), ::expf(v.z));
}

/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE float getByIndex(const float3& v, int i)
{
  return ((float*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE void setByIndex(float3& v, int i, float x)
{
  ((float*)(&v))[i] = x;
}
  
/* float4 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/
CUDAINLINE HOSTDEVICE float4 make_float4(const float s)
{
  return make_float4(s, s, s, s);
}
CUDAINLINE HOSTDEVICE float4 make_float4(const float3& a)
{
  return make_float4(a.x, a.y, a.z, 0.0f);
}
CUDAINLINE HOSTDEVICE float4 make_float4(const int4& a)
{
  return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
CUDAINLINE HOSTDEVICE float4 make_float4(const uint4& a)
{
  return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

CUDAINLINE HOSTDEVICE float3 sqrt(const float3& x) {
    return make_float3(sqrt(x.x), sqrt(x.y), sqrt(x.z));
}

CUDAINLINE HOSTDEVICE bool isnan(const float3& x) {
    return isnan(x.x) || isnan(x.y) || isnan(x.z);
}

CUDAINLINE HOSTDEVICE bool isinf(const float3& x) {
    return isinf(x.x) || isinf(x.y) || isinf(x.z);
}
/** @} */

/** negate */
CUDAINLINE HOSTDEVICE float4 operator-(const float4& a)
{
  return make_float4(-a.x, -a.y, -a.z, -a.w);
}

/** min 
* @{
*/
CUDAINLINE HOSTDEVICE float4 fminf(const float4& a, const float4& b)
{
  return make_float4(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z), fminf(a.w,b.w));
}
CUDAINLINE HOSTDEVICE float fminf(const float4& a)
{
  return fminf(fminf(a.x, a.y), fminf(a.z, a.w));
}
/** @} */

/** max 
* @{
*/
CUDAINLINE HOSTDEVICE float4 fmaxf(const float4& a, const float4& b)
{
  return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
}
CUDAINLINE HOSTDEVICE float fmaxf(const float4& a)
{
  return fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w));
}
/** @} */

/** add 
* @{
*/
CUDAINLINE HOSTDEVICE float4 operator+(const float4& a, const float4& b)
{
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}
CUDAINLINE HOSTDEVICE float4 operator+(const float4& a, const float b)
{
  return make_float4(a.x + b, a.y + b, a.z + b,  a.w + b);
}
CUDAINLINE HOSTDEVICE float4 operator+(const float a, const float4& b)
{
  return make_float4(a + b.x, a + b.y, a + b.z,  a + b.w);
}
CUDAINLINE HOSTDEVICE void operator+=(float4& a, const float4& b)
{
  a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}
/** @} */

/** subtract 
* @{
*/
CUDAINLINE HOSTDEVICE float4 operator-(const float4& a, const float4& b)
{
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
CUDAINLINE HOSTDEVICE float4 operator-(const float4& a, const float b)
{
  return make_float4(a.x - b, a.y - b, a.z - b,  a.w - b);
}
CUDAINLINE HOSTDEVICE float4 operator-(const float a, const float4& b)
{
  return make_float4(a - b.x, a - b.y, a - b.z,  a - b.w);
}
CUDAINLINE HOSTDEVICE void operator-=(float4& a, const float4& b)
{
  a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}
/** @} */

/** multiply 
* @{
*/
CUDAINLINE HOSTDEVICE float4 operator*(const float4& a, const float4& s)
{
  return make_float4(a.x * s.x, a.y * s.y, a.z * s.z, a.w * s.w);
}
CUDAINLINE HOSTDEVICE float4 operator*(const float4& a, const float s)
{
  return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
CUDAINLINE HOSTDEVICE float4 operator*(const float s, const float4& a)
{
  return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
CUDAINLINE HOSTDEVICE void operator*=(float4& a, const float4& s)
{
  a.x *= s.x; a.y *= s.y; a.z *= s.z; a.w *= s.w;
}
CUDAINLINE HOSTDEVICE void operator*=(float4& a, const float s)
{
  a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}
/** @} */

/** divide 
* @{
*/
CUDAINLINE HOSTDEVICE float4 operator/(const float4& a, const float4& b)
{
  return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
CUDAINLINE HOSTDEVICE float4 operator/(const float4& a, const float s)
{
  float inv = 1.0f / s;
  return a * inv;
}
CUDAINLINE HOSTDEVICE float4 operator/(const float s, const float4& a)
{
  return make_float4( s/a.x, s/a.y, s/a.z, s/a.w );
}
CUDAINLINE HOSTDEVICE void operator/=(float4& a, const float s)
{
  float inv = 1.0f / s;
  a *= inv;
}
/** @} */

/** lerp */
CUDAINLINE HOSTDEVICE float4 lerp(const float4& a, const float4& b, const float t)
{
  return a + t*(b-a);
}

/** bilerp */
CUDAINLINE HOSTDEVICE float4 bilerp(const float4& x00, const float4& x10, const float4& x01, const float4& x11,
                                          const float u, const float v)
{
  return lerp( lerp( x00, x10, u ), lerp( x01, x11, u ), v );
}

/** clamp 
* @{
*/
CUDAINLINE HOSTDEVICE float4 clamp(const float4& v, const float a, const float b)
{
  return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

CUDAINLINE HOSTDEVICE float4 clamp(const float4& v, const float4& a, const float4& b)
{
  return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}
/** @} */

/** dot product */
CUDAINLINE HOSTDEVICE float dot(const float4& a, const float4& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

/** length */
CUDAINLINE HOSTDEVICE float length(const float4& r)
{
  return sqrtf(dot(r, r));
}

/** normalize */
CUDAINLINE HOSTDEVICE float4 normalize(const float4& v)
{
  float invLen = 1.0f / sqrtf(dot(v, v));
  return v * invLen;
}

/** floor */
CUDAINLINE HOSTDEVICE float4 floor(const float4& v)
{
  return make_float4(::floorf(v.x), ::floorf(v.y), ::floorf(v.z), ::floorf(v.w));
}

/** reflect */
CUDAINLINE HOSTDEVICE float4 reflect(const float4& i, const float4& n)
{
  return i - 2.0f * n * dot(n,i);
}

/** 
* Faceforward
* Returns N if dot(i, nref) > 0; else -N;
* Typical usage is N = faceforward(N, -ray.dir, N);
* Note that this is opposite of what faceforward does in Cg and GLSL 
*/
CUDAINLINE HOSTDEVICE float4 faceforward(const float4& n, const float4& i, const float4& nref)
{
  return n * copysignf( 1.0f, dot(i, nref) );
}

/** exp */
CUDAINLINE HOSTDEVICE float4 expf(const float4& v)
{
  return make_float4(::expf(v.x), ::expf(v.y), ::expf(v.z), ::expf(v.w));
}

/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE float getByIndex(const float4& v, int i)
{
  return ((float*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE void setByIndex(float4& v, int i, float x)
{
  ((float*)(&v))[i] = x;
}
  
  
/* int functions */
/******************************************************************************/

/** clamp */
CUDAINLINE HOSTDEVICE int clamp(const int f, const int a, const int b)
{
  return max(a, min(f, b));
}

/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE int getByIndex(const int1& v, int i)
{
  return ((int*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE void setByIndex(int1& v, int i, int x)
{
  ((int*)(&v))[i] = x;
}
  

/* int2 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/
CUDAINLINE HOSTDEVICE int2 make_int2(const int s)
{
  return make_int2(s, s);
}
CUDAINLINE HOSTDEVICE int2 make_int2(const float2& a)
{
  return make_int2(int(a.x), int(a.y));
}
/** @} */

/** negate */
CUDAINLINE HOSTDEVICE int2 operator-(const int2& a)
{
  return make_int2(-a.x, -a.y);
}

/** min */
CUDAINLINE HOSTDEVICE int2 min(const int2& a, const int2& b)
{
    return make_int2(min(a.x, b.x), min(a.y, b.y));
}

/** max */
CUDAINLINE HOSTDEVICE int2 max(const int2& a, const int2& b)
{
  return make_int2(max(a.x,b.x), max(a.y,b.y));
}

/** add 
* @{
*/
CUDAINLINE HOSTDEVICE int2 operator+(const int2& a, const int2& b)
{
  return make_int2(a.x + b.x, a.y + b.y);
}
CUDAINLINE HOSTDEVICE void operator+=(int2& a, const int2& b)
{
  a.x += b.x; a.y += b.y;
}
/** @} */

/** subtract 
* @{
*/
CUDAINLINE HOSTDEVICE int2 operator-(const int2& a, const int2& b)
{
  return make_int2(a.x - b.x, a.y - b.y);
}
CUDAINLINE HOSTDEVICE int2 operator-(const int2& a, const int b)
{
  return make_int2(a.x - b, a.y - b);
}
CUDAINLINE HOSTDEVICE void operator-=(int2& a, const int2& b)
{
  a.x -= b.x; a.y -= b.y;
}
/** @} */

/** multiply 
* @{
*/
CUDAINLINE HOSTDEVICE int2 operator*(const int2& a, const int2& b)
{
  return make_int2(a.x * b.x, a.y * b.y);
}
CUDAINLINE HOSTDEVICE int2 operator*(const int2& a, const int s)
{
  return make_int2(a.x * s, a.y * s);
}
CUDAINLINE HOSTDEVICE int2 operator*(const int s, const int2& a)
{
  return make_int2(a.x * s, a.y * s);
}
CUDAINLINE HOSTDEVICE void operator*=(int2& a, const int s)
{
  a.x *= s; a.y *= s;
}
/** @} */

/** clamp 
* @{
*/
CUDAINLINE HOSTDEVICE int2 clamp(const int2& v, const int a, const int b)
{
  return make_int2(clamp(v.x, a, b), clamp(v.y, a, b));
}

CUDAINLINE HOSTDEVICE int2 clamp(const int2& v, const int2& a, const int2& b)
{
  return make_int2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
/** @} */

/** equality 
* @{
*/
CUDAINLINE HOSTDEVICE bool operator==(const int2& a, const int2& b)
{
  return a.x == b.x && a.y == b.y;
}

CUDAINLINE HOSTDEVICE bool operator!=(const int2& a, const int2& b)
{
  return a.x != b.x || a.y != b.y;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE int getByIndex(const int2& v, int i)
{
  return ((int*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE void setByIndex(int2& v, int i, int x)
{
  ((int*)(&v))[i] = x;
}
  

/* int3 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/
CUDAINLINE HOSTDEVICE int3 make_int3(const int s)
{
  return make_int3(s, s, s);
}
CUDAINLINE HOSTDEVICE int3 make_int3(const float3& a)
{
  return make_int3(int(a.x), int(a.y), int(a.z));
}
/** @} */

/** negate */
CUDAINLINE HOSTDEVICE int3 operator-(const int3& a)
{
  return make_int3(-a.x, -a.y, -a.z);
}

/** min */
CUDAINLINE HOSTDEVICE int3 min(const int3& a, const int3& b)
{
  return make_int3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}

/** max */
CUDAINLINE HOSTDEVICE int3 max(const int3& a, const int3& b)
{
  return make_int3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}

/** add 
* @{
*/
CUDAINLINE HOSTDEVICE int3 operator+(const int3& a, const int3& b)
{
  return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
CUDAINLINE HOSTDEVICE void operator+=(int3& a, const int3& b)
{
  a.x += b.x; a.y += b.y; a.z += b.z;
}
/** @} */

/** subtract 
* @{
*/
CUDAINLINE HOSTDEVICE int3 operator-(const int3& a, const int3& b)
{
  return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}

CUDAINLINE HOSTDEVICE void operator-=(int3& a, const int3& b)
{
  a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
/** @} */

/** multiply 
* @{
*/
CUDAINLINE HOSTDEVICE int3 operator*(const int3& a, const int3& b)
{
  return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}
CUDAINLINE HOSTDEVICE int3 operator*(const int3& a, const int s)
{
  return make_int3(a.x * s, a.y * s, a.z * s);
}
CUDAINLINE HOSTDEVICE int3 operator*(const int s, const int3& a)
{
  return make_int3(a.x * s, a.y * s, a.z * s);
}
CUDAINLINE HOSTDEVICE void operator*=(int3& a, const int s)
{
  a.x *= s; a.y *= s; a.z *= s;
}
/** @} */

/** divide 
* @{
*/
CUDAINLINE HOSTDEVICE int3 operator/(const int3& a, const int3& b)
{
  return make_int3(a.x / b.x, a.y / b.y, a.z / b.z);
}
CUDAINLINE HOSTDEVICE int3 operator/(const int3& a, const int s)
{
  return make_int3(a.x / s, a.y / s, a.z / s);
}
CUDAINLINE HOSTDEVICE int3 operator/(const int s, const int3& a)
{
  return make_int3(s /a.x, s / a.y, s / a.z);
}
CUDAINLINE HOSTDEVICE void operator/=(int3& a, const int s)
{
  a.x /= s; a.y /= s; a.z /= s;
}
/** @} */

/** clamp 
* @{
*/
CUDAINLINE HOSTDEVICE int3 clamp(const int3& v, const int a, const int b)
{
  return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

CUDAINLINE HOSTDEVICE int3 clamp(const int3& v, const int3& a, const int3& b)
{
  return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
/** @} */

/** equality 
* @{
*/
CUDAINLINE HOSTDEVICE bool operator==(const int3& a, const int3& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

CUDAINLINE HOSTDEVICE bool operator!=(const int3& a, const int3& b)
{
  return a.x != b.x || a.y != b.y || a.z != b.z;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE int getByIndex(const int3& v, int i)
{
  return ((int*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE void setByIndex(int3& v, int i, int x)
{
  ((int*)(&v))[i] = x;
}
  

/* int4 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/
CUDAINLINE HOSTDEVICE int4 make_int4(const int s)
{
  return make_int4(s, s, s, s);
}
CUDAINLINE HOSTDEVICE int4 make_int4(const float4& a)
{
  return make_int4((int)a.x, (int)a.y, (int)a.z, (int)a.w);
}
/** @} */

/** negate */
CUDAINLINE HOSTDEVICE int4 operator-(const int4& a)
{
  return make_int4(-a.x, -a.y, -a.z, -a.w);
}

/** min */
CUDAINLINE HOSTDEVICE int4 min(const int4& a, const int4& b)
{
  return make_int4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w));
}

/** max */
CUDAINLINE HOSTDEVICE int4 max(const int4& a, const int4& b)
{
  return make_int4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w));
}

/** add 
* @{
*/
CUDAINLINE HOSTDEVICE int4 operator+(const int4& a, const int4& b)
{
  return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
CUDAINLINE HOSTDEVICE void operator+=(int4& a, const int4& b)
{
  a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}
/** @} */

/** subtract 
* @{
*/
CUDAINLINE HOSTDEVICE int4 operator-(const int4& a, const int4& b)
{
  return make_int4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

CUDAINLINE HOSTDEVICE void operator-=(int4& a, const int4& b)
{
  a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}
/** @} */

/** multiply 
* @{
*/
CUDAINLINE HOSTDEVICE int4 operator*(const int4& a, const int4& b)
{
  return make_int4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
CUDAINLINE HOSTDEVICE int4 operator*(const int4& a, const int s)
{
  return make_int4(a.x * s, a.y * s, a.z * s, a.w * s);
}
CUDAINLINE HOSTDEVICE int4 operator*(const int s, const int4& a)
{
  return make_int4(a.x * s, a.y * s, a.z * s, a.w * s);
}
CUDAINLINE HOSTDEVICE void operator*=(int4& a, const int s)
{
  a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}
/** @} */

/** divide 
* @{
*/
CUDAINLINE HOSTDEVICE int4 operator/(const int4& a, const int4& b)
{
  return make_int4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
CUDAINLINE HOSTDEVICE int4 operator/(const int4& a, const int s)
{
  return make_int4(a.x / s, a.y / s, a.z / s, a.w / s);
}
CUDAINLINE HOSTDEVICE int4 operator/(const int s, const int4& a)
{
  return make_int4(s / a.x, s / a.y, s / a.z, s / a.w);
}
CUDAINLINE HOSTDEVICE void operator/=(int4& a, const int s)
{
  a.x /= s; a.y /= s; a.z /= s; a.w /= s;
}
/** @} */

/** clamp 
* @{
*/
CUDAINLINE HOSTDEVICE int4 clamp(const int4& v, const int a, const int b)
{
  return make_int4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

CUDAINLINE HOSTDEVICE int4 clamp(const int4& v, const int4& a, const int4& b)
{
  return make_int4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}
/** @} */

/** equality 
* @{
*/
CUDAINLINE HOSTDEVICE bool operator==(const int4& a, const int4& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

CUDAINLINE HOSTDEVICE bool operator!=(const int4& a, const int4& b)
{
  return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE int getByIndex(const int4& v, int i)
{
  return ((int*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE void setByIndex(int4& v, int i, int x)
{
  ((int*)(&v))[i] = x;
}
  

/* uint functions */
/******************************************************************************/

/** clamp */
CUDAINLINE HOSTDEVICE unsigned int clamp(const unsigned int f, const unsigned int a, const unsigned int b)
{
  return max(a, min(f, b));
}

/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE unsigned int getByIndex(const uint1& v, unsigned int i)
{
  return ((unsigned int*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE void setByIndex(uint1& v, int i, unsigned int x)
{
  ((unsigned int*)(&v))[i] = x;
}
  

/* uint2 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/
CUDAINLINE HOSTDEVICE uint2 make_uint2(const unsigned int s)
{
  return make_uint2(s, s);
}
CUDAINLINE HOSTDEVICE uint2 make_uint2(const float2& a)
{
  return make_uint2((unsigned int)a.x, (unsigned int)a.y);
}
/** @} */

/** min */
CUDAINLINE HOSTDEVICE uint2 min(const uint2& a, const uint2& b)
{
  return make_uint2(min(a.x,b.x), min(a.y,b.y));
}

/** max */
CUDAINLINE HOSTDEVICE uint2 max(const uint2& a, const uint2& b)
{
  return make_uint2(max(a.x,b.x), max(a.y,b.y));
}

/** add
* @{
*/
CUDAINLINE HOSTDEVICE uint2 operator+(const uint2& a, const uint2& b)
{
  return make_uint2(a.x + b.x, a.y + b.y);
}
CUDAINLINE HOSTDEVICE void operator+=(uint2& a, const uint2& b)
{
  a.x += b.x; a.y += b.y;
}
/** @} */

/** subtract
* @{
*/
CUDAINLINE HOSTDEVICE uint2 operator-(const uint2& a, const uint2& b)
{
  return make_uint2(a.x - b.x, a.y - b.y);
}
CUDAINLINE HOSTDEVICE uint2 operator-(const uint2& a, const unsigned int b)
{
  return make_uint2(a.x - b, a.y - b);
}
CUDAINLINE HOSTDEVICE void operator-=(uint2& a, const uint2& b)
{
  a.x -= b.x; a.y -= b.y;
}
/** @} */

/** multiply
* @{
*/
CUDAINLINE HOSTDEVICE uint2 operator*(const uint2& a, const uint2& b)
{
  return make_uint2(a.x * b.x, a.y * b.y);
}
CUDAINLINE HOSTDEVICE uint2 operator*(const uint2& a, const unsigned int s)
{
  return make_uint2(a.x * s, a.y * s);
}
CUDAINLINE HOSTDEVICE uint2 operator*(const unsigned int s, const uint2& a)
{
  return make_uint2(a.x * s, a.y * s);
}
CUDAINLINE HOSTDEVICE void operator*=(uint2& a, const unsigned int s)
{
  a.x *= s; a.y *= s;
}
/** @} */

/** clamp
* @{
*/
CUDAINLINE HOSTDEVICE uint2 clamp(const uint2& v, const unsigned int a, const unsigned int b)
{
  return make_uint2(clamp(v.x, a, b), clamp(v.y, a, b));
}

CUDAINLINE HOSTDEVICE uint2 clamp(const uint2& v, const uint2& a, const uint2& b)
{
  return make_uint2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
/** @} */

/** equality
* @{
*/
CUDAINLINE HOSTDEVICE bool operator==(const uint2& a, const uint2& b)
{
  return a.x == b.x && a.y == b.y;
}

CUDAINLINE HOSTDEVICE bool operator!=(const uint2& a, const uint2& b)
{
  return a.x != b.x || a.y != b.y;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE unsigned int getByIndex(const uint2& v, unsigned int i)
{
  return ((unsigned int*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE void setByIndex(uint2& v, int i, unsigned int x)
{
  ((unsigned int*)(&v))[i] = x;
}
  

/* uint3 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/
CUDAINLINE HOSTDEVICE uint3 make_uint3(const unsigned int s)
{
  return make_uint3(s, s, s);
}
CUDAINLINE HOSTDEVICE uint3 make_uint3(const float3& a)
{
  return make_uint3((unsigned int)a.x, (unsigned int)a.y, (unsigned int)a.z);
}
/** @} */

/** min */
CUDAINLINE HOSTDEVICE uint3 min(const uint3& a, const uint3& b)
{
  return make_uint3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}

/** max */
CUDAINLINE HOSTDEVICE uint3 max(const uint3& a, const uint3& b)
{
  return make_uint3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}

/** add 
* @{
*/
CUDAINLINE HOSTDEVICE uint3 operator+(const uint3& a, const uint3& b)
{
  return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}
CUDAINLINE HOSTDEVICE void operator+=(uint3& a, const uint3& b)
{
  a.x += b.x; a.y += b.y; a.z += b.z;
}
/** @} */

/** subtract
* @{
*/
CUDAINLINE HOSTDEVICE uint3 operator-(const uint3& a, const uint3& b)
{
  return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}

CUDAINLINE HOSTDEVICE void operator-=(uint3& a, const uint3& b)
{
  a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
/** @} */

/** multiply
* @{
*/
CUDAINLINE HOSTDEVICE uint3 operator*(const uint3& a, const uint3& b)
{
  return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}
CUDAINLINE HOSTDEVICE uint3 operator*(const uint3& a, const unsigned int s)
{
  return make_uint3(a.x * s, a.y * s, a.z * s);
}
CUDAINLINE HOSTDEVICE uint3 operator*(const unsigned int s, const uint3& a)
{
  return make_uint3(a.x * s, a.y * s, a.z * s);
}
CUDAINLINE HOSTDEVICE void operator*=(uint3& a, const unsigned int s)
{
  a.x *= s; a.y *= s; a.z *= s;
}
/** @} */

/** divide
* @{
*/
CUDAINLINE HOSTDEVICE uint3 operator/(const uint3& a, const uint3& b)
{
  return make_uint3(a.x / b.x, a.y / b.y, a.z / b.z);
}
CUDAINLINE HOSTDEVICE uint3 operator/(const uint3& a, const unsigned int s)
{
  return make_uint3(a.x / s, a.y / s, a.z / s);
}
CUDAINLINE HOSTDEVICE uint3 operator/(const unsigned int s, const uint3& a)
{
  return make_uint3(s / a.x, s / a.y, s / a.z);
}
CUDAINLINE HOSTDEVICE void operator/=(uint3& a, const unsigned int s)
{
  a.x /= s; a.y /= s; a.z /= s;
}
/** @} */

/** clamp 
* @{
*/
CUDAINLINE HOSTDEVICE uint3 clamp(const uint3& v, const unsigned int a, const unsigned int b)
{
  return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

CUDAINLINE HOSTDEVICE uint3 clamp(const uint3& v, const uint3& a, const uint3& b)
{
  return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
/** @} */

/** equality 
* @{
*/
CUDAINLINE HOSTDEVICE bool operator==(const uint3& a, const uint3& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

CUDAINLINE HOSTDEVICE bool operator!=(const uint3& a, const uint3& b)
{
  return a.x != b.x || a.y != b.y || a.z != b.z;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory 
*/
CUDAINLINE HOSTDEVICE unsigned int getByIndex(const uint3& v, unsigned int i)
{
  return ((unsigned int*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory 
*/
CUDAINLINE HOSTDEVICE void setByIndex(uint3& v, int i, unsigned int x)
{
  ((unsigned int*)(&v))[i] = x;
}
  

/* uint4 functions */
/******************************************************************************/

/** additional constructors 
* @{
*/
CUDAINLINE HOSTDEVICE uint4 make_uint4(const unsigned int s)
{
  return make_uint4(s, s, s, s);
}
CUDAINLINE HOSTDEVICE uint4 make_uint4(const float4& a)
{
  return make_uint4((unsigned int)a.x, (unsigned int)a.y, (unsigned int)a.z, (unsigned int)a.w);
}
/** @} */

/** min
* @{
*/
CUDAINLINE HOSTDEVICE uint4 min(const uint4& a, const uint4& b)
{
  return make_uint4(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z), min(a.w,b.w));
}
/** @} */

/** max 
* @{
*/
CUDAINLINE HOSTDEVICE uint4 max(const uint4& a, const uint4& b)
{
  return make_uint4(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z), max(a.w,b.w));
}
/** @} */

/** add
* @{
*/
CUDAINLINE HOSTDEVICE uint4 operator+(const uint4& a, const uint4& b)
{
  return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
CUDAINLINE HOSTDEVICE void operator+=(uint4& a, const uint4& b)
{
  a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}
/** @} */

/** subtract 
* @{
*/
CUDAINLINE HOSTDEVICE uint4 operator-(const uint4& a, const uint4& b)
{
  return make_uint4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

CUDAINLINE HOSTDEVICE void operator-=(uint4& a, const uint4& b)
{
  a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}
/** @} */

/** multiply
* @{
*/
CUDAINLINE HOSTDEVICE uint4 operator*(const uint4& a, const uint4& b)
{
  return make_uint4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
CUDAINLINE HOSTDEVICE uint4 operator*(const uint4& a, const unsigned int s)
{
  return make_uint4(a.x * s, a.y * s, a.z * s, a.w * s);
}
CUDAINLINE HOSTDEVICE uint4 operator*(const unsigned int s, const uint4& a)
{
  return make_uint4(a.x * s, a.y * s, a.z * s, a.w * s);
}
CUDAINLINE HOSTDEVICE void operator*=(uint4& a, const unsigned int s)
{
  a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}
/** @} */

/** divide 
* @{
*/
CUDAINLINE HOSTDEVICE uint4 operator/(const uint4& a, const uint4& b)
{
  return make_uint4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
CUDAINLINE HOSTDEVICE uint4 operator/(const uint4& a, const unsigned int s)
{
  return make_uint4(a.x / s, a.y / s, a.z / s, a.w / s);
}
CUDAINLINE HOSTDEVICE uint4 operator/(const unsigned int s, const uint4& a)
{
  return make_uint4(s / a.x, s / a.y, s / a.z, s / a.w);
}
CUDAINLINE HOSTDEVICE void operator/=(uint4& a, const unsigned int s)
{
  a.x /= s; a.y /= s; a.z /= s; a.w /= s;
}
/** @} */

/** clamp 
* @{
*/
CUDAINLINE HOSTDEVICE uint4 clamp(const uint4& v, const unsigned int a, const unsigned int b)
{
  return make_uint4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

CUDAINLINE HOSTDEVICE uint4 clamp(const uint4& v, const uint4& a, const uint4& b)
{
  return make_uint4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}
/** @} */

/** equality 
* @{
*/
CUDAINLINE HOSTDEVICE bool operator==(const uint4& a, const uint4& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

CUDAINLINE HOSTDEVICE bool operator!=(const uint4& a, const uint4& b)
{
  return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory 
*/
CUDAINLINE HOSTDEVICE unsigned int getByIndex(const uint4& v, unsigned int i)
{
  return ((unsigned int*)(&v))[i];
}
  
/** If used on the device, this could place the the 'v' in local memory 
*/
CUDAINLINE HOSTDEVICE void setByIndex(uint4& v, int i, unsigned int x)
{
  ((unsigned int*)(&v))[i] = x;
}

/* long long functions */
/******************************************************************************/

/** clamp */
CUDAINLINE HOSTDEVICE long long clamp(const long long f, const long long a, const long long b)
{
    return max(a, min(f, b));
}

/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE long long getByIndex(const longlong1& v, int i)
{
    return ((long long*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE void setByIndex(longlong1& v, int i, long long x)
{
    ((long long*)(&v))[i] = x;
}


/* longlong2 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
CUDAINLINE HOSTDEVICE longlong2 make_longlong2(const long long s)
{
    return make_longlong2(s, s);
}
CUDAINLINE HOSTDEVICE longlong2 make_longlong2(const float2& a)
{
    return make_longlong2(int(a.x), int(a.y));
}
/** @} */

/** negate */
CUDAINLINE HOSTDEVICE longlong2 operator-(const longlong2& a)
{
    return make_longlong2(-a.x, -a.y);
}

/** min */
CUDAINLINE HOSTDEVICE longlong2 min(const longlong2& a, const longlong2& b)
{
    return make_longlong2(min(a.x, b.x), min(a.y, b.y));
}

/** max */
CUDAINLINE HOSTDEVICE longlong2 max(const longlong2& a, const longlong2& b)
{
    return make_longlong2(max(a.x, b.x), max(a.y, b.y));
}

/** add
* @{
*/
CUDAINLINE HOSTDEVICE longlong2 operator+(const longlong2& a, const longlong2& b)
{
    return make_longlong2(a.x + b.x, a.y + b.y);
}
CUDAINLINE HOSTDEVICE void operator+=(longlong2& a, const longlong2& b)
{
    a.x += b.x; a.y += b.y;
}
/** @} */

/** subtract
* @{
*/
CUDAINLINE HOSTDEVICE longlong2 operator-(const longlong2& a, const longlong2& b)
{
    return make_longlong2(a.x - b.x, a.y - b.y);
}
CUDAINLINE HOSTDEVICE longlong2 operator-(const longlong2& a, const long long b)
{
    return make_longlong2(a.x - b, a.y - b);
}
CUDAINLINE HOSTDEVICE void operator-=(longlong2& a, const longlong2& b)
{
    a.x -= b.x; a.y -= b.y;
}
/** @} */

/** multiply
* @{
*/
CUDAINLINE HOSTDEVICE longlong2 operator*(const longlong2& a, const longlong2& b)
{
    return make_longlong2(a.x * b.x, a.y * b.y);
}
CUDAINLINE HOSTDEVICE longlong2 operator*(const longlong2& a, const long long s)
{
    return make_longlong2(a.x * s, a.y * s);
}
CUDAINLINE HOSTDEVICE longlong2 operator*(const long long s, const longlong2& a)
{
    return make_longlong2(a.x * s, a.y * s);
}
CUDAINLINE HOSTDEVICE void operator*=(longlong2& a, const long long s)
{
    a.x *= s; a.y *= s;
}
/** @} */

/** clamp
* @{
*/
CUDAINLINE HOSTDEVICE longlong2 clamp(const longlong2& v, const long long a, const long long b)
{
    return make_longlong2(clamp(v.x, a, b), clamp(v.y, a, b));
}

CUDAINLINE HOSTDEVICE longlong2 clamp(const longlong2& v, const longlong2& a, const longlong2& b)
{
    return make_longlong2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
/** @} */

/** equality
* @{
*/
CUDAINLINE HOSTDEVICE bool operator==(const longlong2& a, const longlong2& b)
{
    return a.x == b.x && a.y == b.y;
}

CUDAINLINE HOSTDEVICE bool operator!=(const longlong2& a, const longlong2& b)
{
    return a.x != b.x || a.y != b.y;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE long long getByIndex(const longlong2& v, int i)
{
    return ((long long*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE void setByIndex(longlong2& v, int i, long long x)
{
    ((long long*)(&v))[i] = x;
}


/* longlong3 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
CUDAINLINE HOSTDEVICE longlong3 make_longlong3(const long long s)
{
    return make_longlong3(s, s, s);
}
CUDAINLINE HOSTDEVICE longlong3 make_longlong3(const float3& a)
{
    return make_longlong3( (long long)a.x, (long long)a.y, (long long)a.z);
}
/** @} */

/** negate */
CUDAINLINE HOSTDEVICE longlong3 operator-(const longlong3& a)
{
    return make_longlong3(-a.x, -a.y, -a.z);
}

/** min */
CUDAINLINE HOSTDEVICE longlong3 min(const longlong3& a, const longlong3& b)
{
    return make_longlong3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

/** max */
CUDAINLINE HOSTDEVICE longlong3 max(const longlong3& a, const longlong3& b)
{
    return make_longlong3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

/** add
* @{
*/
CUDAINLINE HOSTDEVICE longlong3 operator+(const longlong3& a, const longlong3& b)
{
    return make_longlong3(a.x + b.x, a.y + b.y, a.z + b.z);
}
CUDAINLINE HOSTDEVICE void operator+=(longlong3& a, const longlong3& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}
/** @} */

/** subtract
* @{
*/
CUDAINLINE HOSTDEVICE longlong3 operator-(const longlong3& a, const longlong3& b)
{
    return make_longlong3(a.x - b.x, a.y - b.y, a.z - b.z);
}

CUDAINLINE HOSTDEVICE void operator-=(longlong3& a, const longlong3& b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
/** @} */

/** multiply
* @{
*/
CUDAINLINE HOSTDEVICE longlong3 operator*(const longlong3& a, const longlong3& b)
{
    return make_longlong3(a.x * b.x, a.y * b.y, a.z * b.z);
}
CUDAINLINE HOSTDEVICE longlong3 operator*(const longlong3& a, const long long s)
{
    return make_longlong3(a.x * s, a.y * s, a.z * s);
}
CUDAINLINE HOSTDEVICE longlong3 operator*(const long long s, const longlong3& a)
{
    return make_longlong3(a.x * s, a.y * s, a.z * s);
}
CUDAINLINE HOSTDEVICE void operator*=(longlong3& a, const long long s)
{
    a.x *= s; a.y *= s; a.z *= s;
}
/** @} */

/** divide
* @{
*/
CUDAINLINE HOSTDEVICE longlong3 operator/(const longlong3& a, const longlong3& b)
{
    return make_longlong3(a.x / b.x, a.y / b.y, a.z / b.z);
}
CUDAINLINE HOSTDEVICE longlong3 operator/(const longlong3& a, const long long s)
{
    return make_longlong3(a.x / s, a.y / s, a.z / s);
}
CUDAINLINE HOSTDEVICE longlong3 operator/(const long long s, const longlong3& a)
{
    return make_longlong3(s /a.x, s / a.y, s / a.z);
}
CUDAINLINE HOSTDEVICE void operator/=(longlong3& a, const long long s)
{
    a.x /= s; a.y /= s; a.z /= s;
}
/** @} */

/** clamp
* @{
*/
CUDAINLINE HOSTDEVICE longlong3 clamp(const longlong3& v, const long long a, const long long b)
{
    return make_longlong3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

CUDAINLINE HOSTDEVICE longlong3 clamp(const longlong3& v, const longlong3& a, const longlong3& b)
{
    return make_longlong3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
/** @} */

/** equality
* @{
*/
CUDAINLINE HOSTDEVICE bool operator==(const longlong3& a, const longlong3& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

CUDAINLINE HOSTDEVICE bool operator!=(const longlong3& a, const longlong3& b)
{
    return a.x != b.x || a.y != b.y || a.z != b.z;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE long long getByIndex(const longlong3& v, int i)
{
    return ((long long*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE void setByIndex(longlong3& v, int i, int x)
{
    ((long long*)(&v))[i] = x;
}


/* longlong4 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
CUDAINLINE HOSTDEVICE longlong4 make_longlong4(const long long s)
{
    return make_longlong4(s, s, s, s);
}
CUDAINLINE HOSTDEVICE longlong4 make_longlong4(const float4& a)
{
    return make_longlong4((long long)a.x, (long long)a.y, (long long)a.z, (long long)a.w);
}
/** @} */

/** negate */
CUDAINLINE HOSTDEVICE longlong4 operator-(const longlong4& a)
{
    return make_longlong4(-a.x, -a.y, -a.z, -a.w);
}

/** min */
CUDAINLINE HOSTDEVICE longlong4 min(const longlong4& a, const longlong4& b)
{
    return make_longlong4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}

/** max */
CUDAINLINE HOSTDEVICE longlong4 max(const longlong4& a, const longlong4& b)
{
    return make_longlong4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

/** add
* @{
*/
CUDAINLINE HOSTDEVICE longlong4 operator+(const longlong4& a, const longlong4& b)
{
    return make_longlong4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
CUDAINLINE HOSTDEVICE void operator+=(longlong4& a, const longlong4& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}
/** @} */

/** subtract
* @{
*/
CUDAINLINE HOSTDEVICE longlong4 operator-(const longlong4& a, const longlong4& b)
{
    return make_longlong4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

CUDAINLINE HOSTDEVICE void operator-=(longlong4& a, const longlong4& b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}
/** @} */

/** multiply
* @{
*/
CUDAINLINE HOSTDEVICE longlong4 operator*(const longlong4& a, const longlong4& b)
{
    return make_longlong4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
CUDAINLINE HOSTDEVICE longlong4 operator*(const longlong4& a, const long long s)
{
    return make_longlong4(a.x * s, a.y * s, a.z * s, a.w * s);
}
CUDAINLINE HOSTDEVICE longlong4 operator*(const long long s, const longlong4& a)
{
    return make_longlong4(a.x * s, a.y * s, a.z * s, a.w * s);
}
CUDAINLINE HOSTDEVICE void operator*=(longlong4& a, const long long s)
{
    a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}
/** @} */

/** divide
* @{
*/
CUDAINLINE HOSTDEVICE longlong4 operator/(const longlong4& a, const longlong4& b)
{
    return make_longlong4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
CUDAINLINE HOSTDEVICE longlong4 operator/(const longlong4& a, const long long s)
{
    return make_longlong4(a.x / s, a.y / s, a.z / s, a.w / s);
}
CUDAINLINE HOSTDEVICE longlong4 operator/(const long long s, const longlong4& a)
{
    return make_longlong4(s / a.x, s / a.y, s / a.z, s / a.w);
}
CUDAINLINE HOSTDEVICE void operator/=(longlong4& a, const long long s)
{
    a.x /= s; a.y /= s; a.z /= s; a.w /= s;
}
/** @} */

/** clamp
* @{
*/
CUDAINLINE HOSTDEVICE longlong4 clamp(const longlong4& v, const long long a, const long long b)
{
    return make_longlong4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

CUDAINLINE HOSTDEVICE longlong4 clamp(const longlong4& v, const longlong4& a, const longlong4& b)
{
    return make_longlong4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}
/** @} */

/** equality
* @{
*/
CUDAINLINE HOSTDEVICE bool operator==(const longlong4& a, const longlong4& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

CUDAINLINE HOSTDEVICE bool operator!=(const longlong4& a, const longlong4& b)
{
    return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE long long getByIndex(const longlong4& v, int i)
{
    return ((long long*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE void setByIndex(longlong4& v, int i, long long x)
{
    ((long long*)(&v))[i] = x;
}

/* ulonglong functions */
/******************************************************************************/

/** clamp */
CUDAINLINE HOSTDEVICE unsigned long long clamp(const unsigned long long f, const unsigned long long a, const unsigned long long b)
{
    return max(a, min(f, b));
}

/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE unsigned long long getByIndex(const ulonglong1& v, unsigned int i)
{
    return ((unsigned long long*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE void setByIndex(ulonglong1& v, int i, unsigned long long x)
{
    ((unsigned long long*)(&v))[i] = x;
}


/* ulonglong2 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
CUDAINLINE HOSTDEVICE ulonglong2 make_ulonglong2(const unsigned long long s)
{
    return make_ulonglong2(s, s);
}
CUDAINLINE HOSTDEVICE ulonglong2 make_ulonglong2(const float2& a)
{
    return make_ulonglong2((unsigned long long)a.x, (unsigned long long)a.y);
}
/** @} */

/** min */
CUDAINLINE HOSTDEVICE ulonglong2 min(const ulonglong2& a, const ulonglong2& b)
{
    return make_ulonglong2(min(a.x, b.x), min(a.y, b.y));
}

/** max */
CUDAINLINE HOSTDEVICE ulonglong2 max(const ulonglong2& a, const ulonglong2& b)
{
    return make_ulonglong2(max(a.x, b.x), max(a.y, b.y));
}

/** add
* @{
*/
CUDAINLINE HOSTDEVICE ulonglong2 operator+(const ulonglong2& a, const ulonglong2& b)
{
    return make_ulonglong2(a.x + b.x, a.y + b.y);
}
CUDAINLINE HOSTDEVICE void operator+=(ulonglong2& a, const ulonglong2& b)
{
    a.x += b.x; a.y += b.y;
}
/** @} */

/** subtract
* @{
*/
CUDAINLINE HOSTDEVICE ulonglong2 operator-(const ulonglong2& a, const ulonglong2& b)
{
    return make_ulonglong2(a.x - b.x, a.y - b.y);
}
CUDAINLINE HOSTDEVICE ulonglong2 operator-(const ulonglong2& a, const unsigned long long b)
{
    return make_ulonglong2(a.x - b, a.y - b);
}
CUDAINLINE HOSTDEVICE void operator-=(ulonglong2& a, const ulonglong2& b)
{
    a.x -= b.x; a.y -= b.y;
}
/** @} */

/** multiply
* @{
*/
CUDAINLINE HOSTDEVICE ulonglong2 operator*(const ulonglong2& a, const ulonglong2& b)
{
    return make_ulonglong2(a.x * b.x, a.y * b.y);
}
CUDAINLINE HOSTDEVICE ulonglong2 operator*(const ulonglong2& a, const unsigned long long s)
{
    return make_ulonglong2(a.x * s, a.y * s);
}
CUDAINLINE HOSTDEVICE ulonglong2 operator*(const unsigned long long s, const ulonglong2& a)
{
    return make_ulonglong2(a.x * s, a.y * s);
}
CUDAINLINE HOSTDEVICE void operator*=(ulonglong2& a, const unsigned long long s)
{
    a.x *= s; a.y *= s;
}
/** @} */

/** clamp
* @{
*/
CUDAINLINE HOSTDEVICE ulonglong2 clamp(const ulonglong2& v, const unsigned long long a, const unsigned long long b)
{
    return make_ulonglong2(clamp(v.x, a, b), clamp(v.y, a, b));
}

CUDAINLINE HOSTDEVICE ulonglong2 clamp(const ulonglong2& v, const ulonglong2& a, const ulonglong2& b)
{
    return make_ulonglong2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
/** @} */

/** equality
* @{
*/
CUDAINLINE HOSTDEVICE bool operator==(const ulonglong2& a, const ulonglong2& b)
{
    return a.x == b.x && a.y == b.y;
}

CUDAINLINE HOSTDEVICE bool operator!=(const ulonglong2& a, const ulonglong2& b)
{
    return a.x != b.x || a.y != b.y;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE unsigned long long getByIndex(const ulonglong2& v, unsigned int i)
{
    return ((unsigned long long*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory */
CUDAINLINE HOSTDEVICE void setByIndex(ulonglong2& v, int i, unsigned long long x)
{
    ((unsigned long long*)(&v))[i] = x;
}


/* ulonglong3 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
CUDAINLINE HOSTDEVICE ulonglong3 make_ulonglong3(const unsigned long long s)
{
    return make_ulonglong3(s, s, s);
}
CUDAINLINE HOSTDEVICE ulonglong3 make_ulonglong3(const float3& a)
{
    return make_ulonglong3((unsigned long long)a.x, (unsigned long long)a.y, (unsigned long long)a.z);
}
/** @} */

/** min */
CUDAINLINE HOSTDEVICE ulonglong3 min(const ulonglong3& a, const ulonglong3& b)
{
    return make_ulonglong3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

/** max */
CUDAINLINE HOSTDEVICE ulonglong3 max(const ulonglong3& a, const ulonglong3& b)
{
    return make_ulonglong3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

/** add
* @{
*/
CUDAINLINE HOSTDEVICE ulonglong3 operator+(const ulonglong3& a, const ulonglong3& b)
{
    return make_ulonglong3(a.x + b.x, a.y + b.y, a.z + b.z);
}
CUDAINLINE HOSTDEVICE void operator+=(ulonglong3& a, const ulonglong3& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}
/** @} */

/** subtract
* @{
*/
CUDAINLINE HOSTDEVICE ulonglong3 operator-(const ulonglong3& a, const ulonglong3& b)
{
    return make_ulonglong3(a.x - b.x, a.y - b.y, a.z - b.z);
}

CUDAINLINE HOSTDEVICE void operator-=(ulonglong3& a, const ulonglong3& b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
/** @} */

/** multiply
* @{
*/
CUDAINLINE HOSTDEVICE ulonglong3 operator*(const ulonglong3& a, const ulonglong3& b)
{
    return make_ulonglong3(a.x * b.x, a.y * b.y, a.z * b.z);
}
CUDAINLINE HOSTDEVICE ulonglong3 operator*(const ulonglong3& a, const unsigned long long s)
{
    return make_ulonglong3(a.x * s, a.y * s, a.z * s);
}
CUDAINLINE HOSTDEVICE ulonglong3 operator*(const unsigned long long s, const ulonglong3& a)
{
    return make_ulonglong3(a.x * s, a.y * s, a.z * s);
}
CUDAINLINE HOSTDEVICE void operator*=(ulonglong3& a, const unsigned long long s)
{
    a.x *= s; a.y *= s; a.z *= s;
}
/** @} */

/** divide
* @{
*/
CUDAINLINE HOSTDEVICE ulonglong3 operator/(const ulonglong3& a, const ulonglong3& b)
{
    return make_ulonglong3(a.x / b.x, a.y / b.y, a.z / b.z);
}
CUDAINLINE HOSTDEVICE ulonglong3 operator/(const ulonglong3& a, const unsigned long long s)
{
    return make_ulonglong3(a.x / s, a.y / s, a.z / s);
}
CUDAINLINE HOSTDEVICE ulonglong3 operator/(const unsigned long long s, const ulonglong3& a)
{
    return make_ulonglong3(s / a.x, s / a.y, s / a.z);
}
CUDAINLINE HOSTDEVICE void operator/=(ulonglong3& a, const unsigned long long s)
{
    a.x /= s; a.y /= s; a.z /= s;
}
/** @} */

/** clamp
* @{
*/
CUDAINLINE HOSTDEVICE ulonglong3 clamp(const ulonglong3& v, const unsigned long long a, const unsigned long long b)
{
    return make_ulonglong3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

CUDAINLINE HOSTDEVICE ulonglong3 clamp(const ulonglong3& v, const ulonglong3& a, const ulonglong3& b)
{
    return make_ulonglong3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
/** @} */

/** equality
* @{
*/
CUDAINLINE HOSTDEVICE bool operator==(const ulonglong3& a, const ulonglong3& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

CUDAINLINE HOSTDEVICE bool operator!=(const ulonglong3& a, const ulonglong3& b)
{
    return a.x != b.x || a.y != b.y || a.z != b.z;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory
*/
CUDAINLINE HOSTDEVICE unsigned long long getByIndex(const ulonglong3& v, unsigned int i)
{
    return ((unsigned long long*)(&v))[i];
}

/** If used on the device, this could place the the 'v' in local memory
*/
CUDAINLINE HOSTDEVICE void setByIndex(ulonglong3& v, int i, unsigned long long x)
{
    ((unsigned long long*)(&v))[i] = x;
}


/* ulonglong4 functions */
/******************************************************************************/

/** additional constructors
* @{
*/
CUDAINLINE HOSTDEVICE ulonglong4 make_ulonglong4(const unsigned long long s)
{
    return make_ulonglong4(s, s, s, s);
}
CUDAINLINE HOSTDEVICE ulonglong4 make_ulonglong4(const float4& a)
{
    return make_ulonglong4((unsigned long long)a.x, (unsigned long long)a.y, (unsigned long long)a.z, (unsigned long long)a.w);
}
/** @} */

/** min
* @{
*/
CUDAINLINE HOSTDEVICE ulonglong4 min(const ulonglong4& a, const ulonglong4& b)
{
    return make_ulonglong4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}
/** @} */

/** max
* @{
*/
CUDAINLINE HOSTDEVICE ulonglong4 max(const ulonglong4& a, const ulonglong4& b)
{
    return make_ulonglong4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}
/** @} */

/** add
* @{
*/
CUDAINLINE HOSTDEVICE ulonglong4 operator+(const ulonglong4& a, const ulonglong4& b)
{
    return make_ulonglong4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
CUDAINLINE HOSTDEVICE void operator+=(ulonglong4& a, const ulonglong4& b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}
/** @} */

/** subtract
* @{
*/
CUDAINLINE HOSTDEVICE ulonglong4 operator-(const ulonglong4& a, const ulonglong4& b)
{
    return make_ulonglong4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

CUDAINLINE HOSTDEVICE void operator-=(ulonglong4& a, const ulonglong4& b)
{
    a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}
/** @} */

/** multiply
* @{
*/
CUDAINLINE HOSTDEVICE ulonglong4 operator*(const ulonglong4& a, const ulonglong4& b)
{
    return make_ulonglong4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
CUDAINLINE HOSTDEVICE ulonglong4 operator*(const ulonglong4& a, const unsigned long long s)
{
    return make_ulonglong4(a.x * s, a.y * s, a.z * s, a.w * s);
}
CUDAINLINE HOSTDEVICE ulonglong4 operator*(const unsigned long long s, const ulonglong4& a)
{
    return make_ulonglong4(a.x * s, a.y * s, a.z * s, a.w * s);
}
CUDAINLINE HOSTDEVICE void operator*=(ulonglong4& a, const unsigned long long s)
{
    a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}
/** @} */

/** divide
* @{
*/
CUDAINLINE HOSTDEVICE ulonglong4 operator/(const ulonglong4& a, const ulonglong4& b)
{
    return make_ulonglong4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
CUDAINLINE HOSTDEVICE ulonglong4 operator/(const ulonglong4& a, const unsigned long long s)
{
    return make_ulonglong4(a.x / s, a.y / s, a.z / s, a.w / s);
}
CUDAINLINE HOSTDEVICE ulonglong4 operator/(const unsigned long long s, const ulonglong4& a)
{
    return make_ulonglong4(s / a.x, s / a.y, s / a.z, s / a.w);
}
CUDAINLINE HOSTDEVICE void operator/=(ulonglong4& a, const unsigned long long s)
{
    a.x /= s; a.y /= s; a.z /= s; a.w /= s;
}
/** @} */

/** clamp
* @{
*/
CUDAINLINE HOSTDEVICE ulonglong4 clamp(const ulonglong4& v, const unsigned long long a, const unsigned long long b)
{
    return make_ulonglong4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

CUDAINLINE HOSTDEVICE ulonglong4 clamp(const ulonglong4& v, const ulonglong4& a, const ulonglong4& b)
{
    return make_ulonglong4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}
/** @} */

/** equality
* @{
*/
CUDAINLINE HOSTDEVICE bool operator==(const ulonglong4& a, const ulonglong4& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

CUDAINLINE HOSTDEVICE bool operator!=(const ulonglong4& a, const ulonglong4& b)
{
    return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}
/** @} */

/** If used on the device, this could place the the 'v' in local memory
*/
CUDAINLINE HOSTDEVICE unsigned long long getByIndex(const ulonglong4& v, unsigned int i)
{
    return ((unsigned long long*)(&v))[i];
}

/******************************************************************************/

/** Narrowing functions
* @{
*/
CUDAINLINE HOSTDEVICE int2 make_int2(const int3& v0) { return make_int2( v0.x, v0.y ); }
CUDAINLINE HOSTDEVICE int2 make_int2(const int4& v0) { return make_int2( v0.x, v0.y ); }
CUDAINLINE HOSTDEVICE int3 make_int3(const int4& v0) { return make_int3( v0.x, v0.y, v0.z ); }
CUDAINLINE HOSTDEVICE uint2 make_uint2(const uint3& v0) { return make_uint2( v0.x, v0.y ); }
CUDAINLINE HOSTDEVICE uint2 make_uint2(const uint4& v0) { return make_uint2( v0.x, v0.y ); }
CUDAINLINE HOSTDEVICE uint3 make_uint3(const uint4& v0) { return make_uint3( v0.x, v0.y, v0.z ); }
CUDAINLINE HOSTDEVICE longlong2 make_longlong2(const longlong3& v0) { return make_longlong2( v0.x, v0.y ); }
CUDAINLINE HOSTDEVICE longlong2 make_longlong2(const longlong4& v0) { return make_longlong2( v0.x, v0.y ); }
CUDAINLINE HOSTDEVICE longlong3 make_longlong3(const longlong4& v0) { return make_longlong3( v0.x, v0.y, v0.z ); }
CUDAINLINE HOSTDEVICE ulonglong2 make_ulonglong2(const ulonglong3& v0) { return make_ulonglong2( v0.x, v0.y ); }
CUDAINLINE HOSTDEVICE ulonglong2 make_ulonglong2(const ulonglong4& v0) { return make_ulonglong2( v0.x, v0.y ); }
CUDAINLINE HOSTDEVICE ulonglong3 make_ulonglong3(const ulonglong4& v0) { return make_ulonglong3( v0.x, v0.y, v0.z ); }
CUDAINLINE HOSTDEVICE float2 make_float2(const float3& v0) { return make_float2( v0.x, v0.y ); }
CUDAINLINE HOSTDEVICE float2 make_float2(const float4& v0) { return make_float2( v0.x, v0.y ); }
CUDAINLINE HOSTDEVICE float3 make_float3(const float4& v0) { return make_float3( v0.x, v0.y, v0.z ); }
/** @} */

/** Assemble functions from smaller vectors 
* @{
*/
CUDAINLINE HOSTDEVICE int3 make_int3(const int v0, const int2& v1) { return make_int3( v0, v1.x, v1.y ); }
CUDAINLINE HOSTDEVICE int3 make_int3(const int2& v0, const int v1) { return make_int3( v0.x, v0.y, v1 ); }
CUDAINLINE HOSTDEVICE int4 make_int4(const int v0, const int v1, const int2& v2) { return make_int4( v0, v1, v2.x, v2.y ); }
CUDAINLINE HOSTDEVICE int4 make_int4(const int v0, const int2& v1, const int v2) { return make_int4( v0, v1.x, v1.y, v2 ); }
CUDAINLINE HOSTDEVICE int4 make_int4(const int2& v0, const int v1, const int v2) { return make_int4( v0.x, v0.y, v1, v2 ); }
CUDAINLINE HOSTDEVICE int4 make_int4(const int v0, const int3& v1) { return make_int4( v0, v1.x, v1.y, v1.z ); }
CUDAINLINE HOSTDEVICE int4 make_int4(const int3& v0, const int v1) { return make_int4( v0.x, v0.y, v0.z, v1 ); }
CUDAINLINE HOSTDEVICE int4 make_int4(const int2& v0, const int2& v1) { return make_int4( v0.x, v0.y, v1.x, v1.y ); }
CUDAINLINE HOSTDEVICE uint3 make_uint3(const unsigned int v0, const uint2& v1) { return make_uint3( v0, v1.x, v1.y ); }
CUDAINLINE HOSTDEVICE uint3 make_uint3(const uint2& v0, const unsigned int v1) { return make_uint3( v0.x, v0.y, v1 ); }
CUDAINLINE HOSTDEVICE uint4 make_uint4(const unsigned int v0, const unsigned int v1, const uint2& v2) { return make_uint4( v0, v1, v2.x, v2.y ); }
CUDAINLINE HOSTDEVICE uint4 make_uint4(const unsigned int v0, const uint2& v1, const unsigned int v2) { return make_uint4( v0, v1.x, v1.y, v2 ); }
CUDAINLINE HOSTDEVICE uint4 make_uint4(const uint2& v0, const unsigned int v1, const unsigned int v2) { return make_uint4( v0.x, v0.y, v1, v2 ); }
CUDAINLINE HOSTDEVICE uint4 make_uint4(const unsigned int v0, const uint3& v1) { return make_uint4( v0, v1.x, v1.y, v1.z ); }
CUDAINLINE HOSTDEVICE uint4 make_uint4(const uint3& v0, const unsigned int v1) { return make_uint4( v0.x, v0.y, v0.z, v1 ); }
CUDAINLINE HOSTDEVICE uint4 make_uint4(const uint2& v0, const uint2& v1) { return make_uint4( v0.x, v0.y, v1.x, v1.y ); }
CUDAINLINE HOSTDEVICE longlong3 make_longlong3(const long long v0, const longlong2& v1) { return make_longlong3(v0, v1.x, v1.y); }
CUDAINLINE HOSTDEVICE longlong3 make_longlong3(const longlong2& v0, const long long v1) { return make_longlong3(v0.x, v0.y, v1); }
CUDAINLINE HOSTDEVICE longlong4 make_longlong4(const long long v0, const long long v1, const longlong2& v2) { return make_longlong4(v0, v1, v2.x, v2.y); }
CUDAINLINE HOSTDEVICE longlong4 make_longlong4(const long long v0, const longlong2& v1, const long long v2) { return make_longlong4(v0, v1.x, v1.y, v2); }
CUDAINLINE HOSTDEVICE longlong4 make_longlong4(const longlong2& v0, const long long v1, const long long v2) { return make_longlong4(v0.x, v0.y, v1, v2); }
CUDAINLINE HOSTDEVICE longlong4 make_longlong4(const long long v0, const longlong3& v1) { return make_longlong4(v0, v1.x, v1.y, v1.z); }
CUDAINLINE HOSTDEVICE longlong4 make_longlong4(const longlong3& v0, const long long v1) { return make_longlong4(v0.x, v0.y, v0.z, v1); }
CUDAINLINE HOSTDEVICE longlong4 make_longlong4(const longlong2& v0, const longlong2& v1) { return make_longlong4(v0.x, v0.y, v1.x, v1.y); }
CUDAINLINE HOSTDEVICE ulonglong3 make_ulonglong3(const unsigned long long v0, const ulonglong2& v1) { return make_ulonglong3(v0, v1.x, v1.y); }
CUDAINLINE HOSTDEVICE ulonglong3 make_ulonglong3(const ulonglong2& v0, const unsigned long long v1) { return make_ulonglong3(v0.x, v0.y, v1); }
CUDAINLINE HOSTDEVICE ulonglong4 make_ulonglong4(const unsigned long long v0, const unsigned long long v1, const ulonglong2& v2) { return make_ulonglong4(v0, v1, v2.x, v2.y); }
CUDAINLINE HOSTDEVICE ulonglong4 make_ulonglong4(const unsigned long long v0, const ulonglong2& v1, const unsigned long long v2) { return make_ulonglong4(v0, v1.x, v1.y, v2); }
CUDAINLINE HOSTDEVICE ulonglong4 make_ulonglong4(const ulonglong2& v0, const unsigned long long v1, const unsigned long long v2) { return make_ulonglong4(v0.x, v0.y, v1, v2); }
CUDAINLINE HOSTDEVICE ulonglong4 make_ulonglong4(const unsigned long long v0, const ulonglong3& v1) { return make_ulonglong4(v0, v1.x, v1.y, v1.z); }
CUDAINLINE HOSTDEVICE ulonglong4 make_ulonglong4(const ulonglong3& v0, const unsigned long long v1) { return make_ulonglong4(v0.x, v0.y, v0.z, v1); }
CUDAINLINE HOSTDEVICE ulonglong4 make_ulonglong4(const ulonglong2& v0, const ulonglong2& v1) { return make_ulonglong4(v0.x, v0.y, v1.x, v1.y); }
CUDAINLINE HOSTDEVICE float3 make_float3(const float2& v0, const float v1) { return make_float3(v0.x, v0.y, v1); }
CUDAINLINE HOSTDEVICE float3 make_float3(const float v0, const float2& v1) { return make_float3( v0, v1.x, v1.y ); }
CUDAINLINE HOSTDEVICE float4 make_float4(const float v0, const float v1, const float2& v2) { return make_float4( v0, v1, v2.x, v2.y ); }
CUDAINLINE HOSTDEVICE float4 make_float4(const float v0, const float2& v1, const float v2) { return make_float4( v0, v1.x, v1.y, v2 ); }
CUDAINLINE HOSTDEVICE float4 make_float4(const float2& v0, const float v1, const float v2) { return make_float4( v0.x, v0.y, v1, v2 ); }
CUDAINLINE HOSTDEVICE float4 make_float4(const float v0, const float3& v1) { return make_float4( v0, v1.x, v1.y, v1.z ); }
CUDAINLINE HOSTDEVICE float4 make_float4(const float3& v0, const float v1) { return make_float4( v0.x, v0.y, v0.z, v1 ); }
CUDAINLINE HOSTDEVICE float4 make_float4(const float2& v0, const float2& v1) { return make_float4( v0.x, v0.y, v1.x, v1.y ); }

__forceinline__ __device__ float3 toSRGB(const float3& c)
{
    float  invGamma = 1.0f / 2.4f;
    float3 powed = make_float3(powf(c.x, invGamma), powf(c.y, invGamma), powf(c.z, invGamma));
    return make_float3(
        c.x < 0.0031308f ? 12.92f * c.x : 1.055f * powed.x - 0.055f,
        c.y < 0.0031308f ? 12.92f * c.y : 1.055f * powed.y - 0.055f,
        c.z < 0.0031308f ? 12.92f * c.z : 1.055f * powed.z - 0.055f);
}

//__forceinline__ __device__ float dequantizeUnsigned8Bits( const unsigned char i )
//{
//    enum { N = (1 << 8) - 1 };
//    return min((float)i / (float)N), 1.f)
//}
__forceinline__ __device__ unsigned char quantizeUnsigned8Bits(float x)
{
    x = clamp(x, 0.0f, 1.0f);
    enum { N = (1 << 8) - 1, Np1 = (1 << 8) };
    return (unsigned char)min((unsigned int)(x * (float)Np1), (unsigned int)N);
}

HOSTDEVICE CUDAINLINE float3 gammaCorrect(float3 c) {
    float gamma = 2.2;
    return pow(c, make_float3(1.0f / gamma));
}

__forceinline__ __device__ uchar4 make_color(const float3& c)
{
    // first apply gamma, then convert to unsigned char
    float3 srgb = gammaCorrect(clamp(c, 0.0f, 1.0f));
    return make_uchar4(quantizeUnsigned8Bits(srgb.x), quantizeUnsigned8Bits(srgb.y), quantizeUnsigned8Bits(srgb.z), 255u);
}
__forceinline__ __device__ uchar4 make_color(const float4& c)
{
    return make_color(make_float3(c.x, c.y, c.z));
}

__forceinline__ __device__ bool isfinite(const float3& c) {
    return isfinite(c.x) && isfinite(c.y) && isfinite(c.z);
}

/** @} */

struct mat4 {
    float m[4][4];

    __host__ __device__ __forceinline__ mat4() {
        m[0][0] = 1.0; m[1][0] = 0.0; m[2][0] = 0.0; m[3][0] = 0.0;
        m[0][1] = 0.0; m[1][1] = 1.0; m[2][1] = 0.0; m[3][1] = 0.0;
        m[0][2] = 0.0; m[1][2] = 0.0; m[2][2] = 1.0; m[3][2] = 0.0;
        m[0][3] = 0.0; m[1][3] = 0.0; m[2][3] = 0.0; m[3][3] = 1.0;
    }

    __host__ __device__ __forceinline__ mat4(
        const float m11, const float m12, const float m13, const float m14,
        const float m21, const float m22, const float m23, const float m24,
        const float m31, const float m32, const float m33, const float m34,
        const float m41, const float m42, const float m43, const float m44
    ) {
        m[0][0] = m11; m[1][0] = m12; m[2][0] = m13; m[3][0] = m14;
        m[0][1] = m21; m[1][1] = m22; m[2][1] = m23; m[3][1] = m24;
        m[0][2] = m31; m[1][2] = m32; m[2][2] = m33; m[3][2] = m34;
        m[0][3] = m41; m[1][3] = m42; m[2][3] = m43; m[3][3] = m44;
    }

    __host__ __device__ __forceinline__ mat4(float* data) {
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 4; ++j) {
                m[j][i] = data[i * 4 + j];
            }
        }
    }

    __host__ __device__ __forceinline__ float* operator[] (const size_t idx) {
        return m[idx];
    }

    __host__ __device__ __forceinline__ float4 operator*(const float4& v) const {
        float4 ret;
        ret.x = m[0][0] * v.x + m[1][0] * v.y + m[2][0] * v.z + m[3][0] * v.w;
        ret.y = m[0][1] * v.x + m[1][1] * v.y + m[2][1] * v.z + m[3][1] * v.w;
        ret.z = m[0][2] * v.x + m[1][2] * v.y + m[2][2] * v.z + m[3][2] * v.w;
        ret.w = m[0][3] * v.x + m[1][3] * v.y + m[2][3] * v.z + m[3][3] * v.w;
        return ret;
    }

    __host__ __device__ __forceinline__ mat4 operator*(const float f) const {
        mat4 ret;
        ret[0][0] = m[0][0] * f; ret[1][0] = m[1][0] * f; ret[2][0] = m[2][0] * f; ret[3][0] = m[3][0] * f;
        ret[0][1] = m[0][1] * f; ret[1][1] = m[1][1] * f; ret[2][1] = m[2][1] * f; ret[3][1] = m[3][1] * f;
        ret[0][2] = m[0][2] * f; ret[1][2] = m[1][2] * f; ret[2][2] = m[2][2] * f; ret[3][2] = m[3][2] * f;
        ret[0][3] = m[0][3] * f; ret[1][3] = m[1][3] * f; ret[2][3] = m[2][3] * f; ret[3][3] = m[3][3] * f;
        return ret;
    }

    __host__ __device__ __forceinline__ mat4 operator/(const float f) const {
        mat4 ret;
        ret[0][0] = m[0][0] / f; ret[1][0] = m[1][0] / f; ret[2][0] = m[2][0] / f; ret[3][0] = m[3][0] / f;
        ret[0][1] = m[0][1] / f; ret[1][1] = m[1][1] / f; ret[2][1] = m[2][1] / f; ret[3][1] = m[3][1] / f;
        ret[0][2] = m[0][2] / f; ret[1][2] = m[1][2] / f; ret[2][2] = m[2][2] / f; ret[3][2] = m[3][2] / f;
        ret[0][3] = m[0][3] / f; ret[1][3] = m[1][3] / f; ret[2][3] = m[2][3] / f; ret[3][3] = m[3][3] / f;
        return ret;
    }

    __host__ __device__ __forceinline__ mat4 operator+(const mat4& other) const {
        mat4 ret;
        ret[0][0] = m[0][0] + other.m[0][0]; ret[1][0] = m[1][0] + other.m[1][0]; ret[2][0] = m[2][0] + other.m[2][0]; ret[3][0] = m[3][0] + other.m[3][0];
        ret[0][1] = m[0][1] + other.m[0][1]; ret[1][1] = m[1][1] + other.m[1][1]; ret[2][1] = m[2][1] + other.m[2][1]; ret[3][1] = m[3][1] + other.m[3][1];
        ret[0][2] = m[0][2] + other.m[0][2]; ret[1][2] = m[1][2] + other.m[1][2]; ret[2][2] = m[2][2] + other.m[2][2]; ret[3][2] = m[3][2] + other.m[3][2];
        ret[0][3] = m[0][3] + other.m[0][3]; ret[1][3] = m[1][3] + other.m[1][3]; ret[2][3] = m[2][3] + other.m[2][3]; ret[3][3] = m[3][3] + other.m[3][3];
        return ret;
    }

    __host__ __device__ __forceinline__ mat4 operator-(const mat4& other) const {
        mat4 ret;
        ret[0][0] = m[0][0] - other.m[0][0]; ret[1][0] = m[1][0] - other.m[1][0]; ret[2][0] = m[2][0] - other.m[2][0]; ret[3][0] = m[3][0] - other.m[3][0];
        ret[0][1] = m[0][1] - other.m[0][1]; ret[1][1] = m[1][1] - other.m[1][1]; ret[2][1] = m[2][1] - other.m[2][1]; ret[3][1] = m[3][1] - other.m[3][1];
        ret[0][2] = m[0][2] - other.m[0][2]; ret[1][2] = m[1][2] - other.m[1][2]; ret[2][2] = m[2][2] - other.m[2][2]; ret[3][2] = m[3][2] - other.m[3][2];
        ret[0][3] = m[0][3] - other.m[0][3]; ret[1][3] = m[1][3] - other.m[1][3]; ret[2][3] = m[2][3] - other.m[2][3]; ret[3][3] = m[3][3] - other.m[3][3];
        return ret;
    }

    __host__ __device__ __forceinline__ mat4 operator*(const mat4& other) const {
        auto a11 = m[0][0], a12 = m[1][0], a13 = m[2][0], a14 = m[3][0];
        auto a21 = m[0][1], a22 = m[1][1], a23 = m[2][1], a24 = m[3][1];
        auto a31 = m[0][2], a32 = m[1][2], a33 = m[2][2], a34 = m[3][2];
        auto a41 = m[0][3], a42 = m[1][3], a43 = m[2][3], a44 = m[3][3];

        auto b11 = other.m[0][0], b12 = other.m[1][0], b13 = other.m[2][0], b14 = other.m[3][0];
        auto b21 = other.m[0][1], b22 = other.m[1][1], b23 = other.m[2][1], b24 = other.m[3][1];
        auto b31 = other.m[0][2], b32 = other.m[1][2], b33 = other.m[2][2], b34 = other.m[3][2];
        auto b41 = other.m[0][3], b42 = other.m[1][3], b43 = other.m[2][3], b44 = other.m[3][3];

        mat4 ret;
        ret[0][0] = a11 * b11 + a12 * b21 + a13 * b31 + a14 * b41;
        ret[0][1] = a11 * b12 + a12 * b22 + a13 * b32 + a14 * b42;
        ret[0][2] = a11 * b13 + a12 * b23 + a13 * b33 + a14 * b43;
        ret[0][3] = a11 * b14 + a12 * b24 + a13 * b34 + a14 * b44;

        ret[1][0] = a21 * b11 + a22 * b21 + a23 * b31 + a24 * b41;
        ret[1][1] = a21 * b12 + a22 * b22 + a23 * b32 + a24 * b42;
        ret[1][2] = a21 * b13 + a22 * b23 + a23 * b33 + a24 * b43;
        ret[1][3] = a21 * b14 + a22 * b24 + a23 * b34 + a24 * b44;

        ret[2][0] = a31 * b11 + a32 * b21 + a33 * b31 + a34 * b41;
        ret[2][1] = a31 * b12 + a32 * b22 + a33 * b32 + a34 * b42;
        ret[2][2] = a31 * b13 + a32 * b23 + a33 * b33 + a34 * b43;
        ret[2][3] = a31 * b14 + a32 * b24 + a33 * b34 + a34 * b44;

        ret[3][0] = a41 * b11 + a42 * b21 + a43 * b31 + a44 * b41;
        ret[3][1] = a41 * b12 + a42 * b22 + a43 * b32 + a44 * b42;
        ret[3][2] = a41 * b13 + a42 * b23 + a43 * b33 + a44 * b43;
        ret[3][3] = a41 * b14 + a42 * b24 + a43 * b34 + a44 * b44;
        return ret;
    }

    __host__ __device__ __forceinline__ mat4 transpose() const {
        mat4 ret;
        ret[0][0] = m[0][0]; ret[0][1] = m[1][0]; ret[0][2] = m[2][0]; ret[0][3] = m[3][0];
        ret[1][0] = m[0][1]; ret[1][1] = m[1][1]; ret[1][2] = m[2][1]; ret[1][3] = m[3][1];
        ret[2][0] = m[0][2]; ret[2][1] = m[1][2]; ret[2][2] = m[2][2]; ret[2][3] = m[3][2];
        ret[3][0] = m[0][3]; ret[3][1] = m[1][3]; ret[3][2] = m[2][3]; ret[3][3] = m[3][3];
        return ret;
    }

    __host__ __device__ __forceinline__ float det() const {
        auto n11 = m[0][0], n12 = m[1][0], n13 = m[2][0], n14 = m[3][0];
        auto n21 = m[0][1], n22 = m[1][1], n23 = m[2][1], n24 = m[3][1];
        auto n31 = m[0][2], n32 = m[1][2], n33 = m[2][2], n34 = m[3][2];
        auto n41 = m[0][3], n42 = m[1][3], n43 = m[2][3], n44 = m[3][3];

        return (
            n41 * (
                +n14 * n23 * n32
                - n13 * n24 * n32
                - n14 * n22 * n33
                + n12 * n24 * n33
                + n13 * n22 * n34
                - n12 * n23 * n34
                ) +
            n42 * (
                +n11 * n23 * n34
                - n11 * n24 * n33
                + n14 * n21 * n33
                - n13 * n21 * n34
                + n13 * n24 * n31
                - n14 * n23 * n31
                ) +
            n43 * (
                +n11 * n24 * n32
                - n11 * n22 * n34
                - n14 * n21 * n32
                + n12 * n21 * n34
                + n14 * n22 * n31
                - n12 * n24 * n31
                ) +
            n44 * (
                -n13 * n22 * n31
                - n11 * n23 * n32
                + n11 * n22 * n33
                + n13 * n21 * n32
                - n12 * n21 * n33
                + n12 * n23 * n31
                )
            );
    }

    __host__ __device__ __forceinline__ mat4 inverse() const {
        auto n11 = m[0][0], n12 = m[1][0], n13 = m[2][0], n14 = m[3][0];
        auto n21 = m[0][1], n22 = m[1][1], n23 = m[2][1], n24 = m[3][1];
        auto n31 = m[0][2], n32 = m[1][2], n33 = m[2][2], n34 = m[3][2];
        auto n41 = m[0][3], n42 = m[1][3], n43 = m[2][3], n44 = m[3][3];

        auto t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
        auto t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
        auto t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
        auto t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;

        auto det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
        auto idet = 1.0f / det;

        mat4 ret;

        ret[0][0] = t11 * idet;
        ret[0][1] = (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44 - n21 * n33 * n44) * idet;
        ret[0][2] = (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44 + n21 * n32 * n44) * idet;
        ret[0][3] = (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43 - n21 * n32 * n43) * idet;

        ret[1][0] = t12 * idet;
        ret[1][1] = (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44 + n11 * n33 * n44) * idet;
        ret[1][2] = (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44 - n11 * n32 * n44) * idet;
        ret[1][3] = (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43 + n11 * n32 * n43) * idet;

        ret[2][0] = t13 * idet;
        ret[2][1] = (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44 - n11 * n23 * n44) * idet;
        ret[2][2] = (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44 + n11 * n22 * n44) * idet;
        ret[2][3] = (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43 - n11 * n22 * n43) * idet;

        ret[3][0] = t14 * idet;
        ret[3][1] = (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34 + n11 * n23 * n34) * idet;
        ret[3][2] = (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34 - n11 * n22 * n34) * idet;
        ret[3][3] = (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33 + n11 * n22 * n33) * idet;

        return ret;
    }

    __host__ __device__ __forceinline__ void zero() {
        m[0][0] = 0.0; m[1][0] = 0.0; m[2][0] = 0.0; m[3][0] = 0.0;
        m[0][1] = 0.0; m[1][1] = 0.0; m[2][1] = 0.0; m[3][1] = 0.0;
        m[0][2] = 0.0; m[1][2] = 0.0; m[2][2] = 0.0; m[3][2] = 0.0;
        m[0][3] = 0.0; m[1][3] = 0.0; m[2][3] = 0.0; m[3][3] = 0.0;
    }

    __host__ __device__ __forceinline__ void identity() {
        m[0][0] = 1.0; m[1][0] = 0.0; m[2][0] = 0.0; m[3][0] = 0.0;
        m[0][1] = 0.0; m[1][1] = 1.0; m[2][1] = 0.0; m[3][1] = 0.0;
        m[0][2] = 0.0; m[1][2] = 0.0; m[2][2] = 1.0; m[3][2] = 0.0;
        m[0][3] = 0.0; m[1][3] = 0.0; m[2][3] = 0.0; m[3][3] = 1.0;
    }

    __host__ __device__ __forceinline__ mat4& operator*=(const float f) { return *this = *this * f; }
    __host__ __device__ __forceinline__ mat4& operator/=(const float f) { return *this = *this / f; }
    __host__ __device__ __forceinline__ mat4& operator+=(const mat4& m) { return *this = *this + m; }
    __host__ __device__ __forceinline__ mat4& operator-=(const mat4& m) { return *this = *this - m; }
    __host__ __device__ __forceinline__ mat4& operator*=(const mat4& m) { return *this = *this * m; }

};
