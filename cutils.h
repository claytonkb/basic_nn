//cutils.h

#ifndef CUTILS_H
#define CUTILS_H

//#include <stdio.h>
//#include <string.h>
//#include <stdlib.h>
//#include <stddef.h>
//#include <assert.h>
//#include <setjmp.h>
//#include <stdarg.h>
//#include <time.h>
//#include <stdint.h>
//#include <inttypes.h>
//#include <time.h>
//#include <unistd.h>
//#include <math.h>

#include <stddef.h>
#include <stdio.h>
#include <memory.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

/*****************************************************************************
 *                                                                           *
 *                            MISC C UTILITIES                               *
 *                                                                           *
 ****************************************************************************/

#define _die            do{ fprintf(stderr, "Died at %s line %d\n", __FILE__, __LINE__); exit(1); } while(0) // die#
#define _fatal(x)       do{ fprintf(stderr, "FATAL: %s in %s()\n", x, __func__); _die; } while(0) // _fatal#
#define _warn(x)        do{ fprintf(stderr, "WARNING: %s in %s() at %s line %d\n", x, __func__, __FILE__, __LINE__); fflush(stderr); } while(0) // warn#
#define _error(x)       do{ fprintf(stderr, "ERROR: %s in %s() at %s line %d\n", x, __func__, __FILE__, __LINE__); fflush(stderr); } while(0) // error#
#define _pigs_fly       _fatal("Pigs CAN fly...") // _pigs_fly#

#define _d(x)           do{ fprintf(stderr, "%s %x\n", #x, (unsigned)x); fflush(stderr); } while(0)
#define _dw(x)          do{ fprintf(stderr, "%08x ", (unsigned)x); fflush(stderr); } while(0)
#define _dc(x)          do{ fprintf(stderr, "%02x ",  (uint8_t)x); fflush(stderr); } while(0)
#define _dd(x)          do{ fprintf(stderr, "%s %d\n", #x, (unsigned)x); } while(0) // dd#
#define _df(x)          do{ fprintf(stderr, "%s %f\n", #x, (float)x); } while(0) // df#
#define _du(x)          do{ fprintf(stderr, "%s %u\n", #x, (unsigned)x); } while(0) // du#

#define _h(x)           do{ fprintf(stderr, "%08x", (unsigned)x); fflush(stderr); } while(0)
#define _hc(x)          do{ fprintf(stderr, "%02x",  (uint8_t)x); fflush(stderr); } while(0)

#define _endl           do{ fprintf(stderr, "\n"); fflush(stderr); } while(0)
#define _prn(x)         do{ fprintf(stderr, "%s", x); fflush(stderr); } while(0)
#define _say(x)         do{ fprintf(stderr, "%s\n", x); fflush(stderr); } while(0)  // _say#
#define _trace          do{ fprintf(stderr, "TRACE: %s() in %s line %d\n", __func__, __FILE__, __LINE__); fflush(stderr); } while(0)   // _trace# 

#endif // CUTILS_H


