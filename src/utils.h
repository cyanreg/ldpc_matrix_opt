/*
 * This file is part of ldpc_matrix_opt.
 *
 * ldpc_matrix_opt is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * ldpc_matrix_opt is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with ldpc_matrix_opt; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <time.h>

static inline double diff_timespec(const struct timespec *time1,
                                   const struct timespec *time0)
{
    return (time1->tv_sec - time0->tv_sec) +
           (time1->tv_nsec - time0->tv_nsec) / 1000000000.0;
}
