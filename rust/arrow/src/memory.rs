// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Defines memory-related functions, such as allocate/deallocate/reallocate memory
//! regions.
//!
//! These functions allocate memory with [`ALIGNMENT`]-byte alignment.
//! Additionally, they are easier to use correctly because they safely handle
//! zero-size allocations and abort on errors (instead of returning a null
//! pointer).

use std::alloc::Layout;
use std::mem::align_of;
use std::ptr::NonNull;

/// Alignment for allocations made by functions in this module.
// Note: The implementation assumes that `ALIGNMENT` is nonzero and a power of
// two. If either of these things change, the implementation must be updated.
pub const ALIGNMENT: usize = 64;

/// Panics if `size`, when rounded up to the nearest multiple of [`ALIGNMENT`],
/// overflows.
fn check_size(size: usize) {
    // This check comes from the implementation of `Layout::from_size_align`.
    if size > std::usize::MAX - (ALIGNMENT - 1) {
        panic!("size overflow");
    }
}

/// Returns `Layout` with alignment [`ALIGNMENT`].
///
/// # Panics
///
/// Panics if `size`, when rounded up to the nearest multiple of [`ALIGNMENT`],
/// overflows.
fn layout_aligned(size: usize) -> Layout {
    check_size(size);
    unsafe { Layout::from_size_align_unchecked(size, ALIGNMENT) }
}

/// Returns a `NonNull<u8>` aligned to [`ALIGNMENT`].
fn nonnull_aligned() -> NonNull<u8> {
    // This is equivalent to the implementation of
    // `NonNull::<T>::dangling().cast::<u8>()` where `T` is some 64-byte
    // aligned type.
    unsafe { NonNull::new_unchecked(ALIGNMENT as *mut u8) }
}

/// Allocates a block of memory of the specified size with [`ALIGNMENT`]-byte
/// alignment.
///
/// # Panics
///
/// Panics if `size`, when rounded up to the nearest multiple of [`ALIGNMENT`],
/// overflows.
///
/// # Aborts
///
/// Aborts if allocation fails.
pub fn allocate_aligned(size: usize) -> NonNull<u8> {
    // If the `size` is zero, we must not call `alloc`.
    if size == 0 {
        nonnull_aligned()
    } else {
        let layout = layout_aligned(size);
        let ptr = unsafe { std::alloc::alloc(layout) };
        // The allocator indicates an error by returning a null pointer.
        if ptr.is_null() {
            // Abort on error.
            std::alloc::handle_alloc_error(layout)
        } else {
            unsafe { NonNull::new_unchecked(ptr) }
        }
    }
}

/// Deallocates the block of memory.
///
/// # Safety
///
/// The caller must ensure all of the following:
///
/// * `ptr` must denote a block of memory allocated by `allocate_aligned`/`reallocate`
/// * `size` must be the size that was used by `allocate_aligned`/`reallocate`
///
/// # Panics
///
/// Panics if `size`, when rounded up to the nearest multiple of [`ALIGNMENT`],
/// overflows. (This should never happen because
/// `allocate_aligned`/`reallocate` panic in this case.)
pub unsafe fn free_aligned(ptr: NonNull<u8>, size: usize) {
    if size != 0 {
        std::alloc::dealloc(ptr.as_ptr(), layout_aligned(size));
    }
}

/// Shrinks or grows a block of memory to the `new_size`.
///
/// # Safety
///
/// The caller must ensure all of the following:
///
/// * `ptr` must denote a block of memory allocated by `allocate_aligned`/`reallocate`
/// * `old_size` must be the size that was used by `allocate_aligned`/`reallocate`
///
/// # Panics
///
/// Panics if `new_size`, when rounded up to the nearest multiple of
/// [`ALIGNMENT`], overflows.
///
/// # Aborts
///
/// Aborts if allocation fails.
pub unsafe fn reallocate(ptr: NonNull<u8>, old_size: usize, new_size: usize) -> NonNull<u8> {
    if new_size == 0 {
        // If the `new_size` is zero, we must not call `realloc`. We just free
        // the old memory and return an aligned pointer.
        free_aligned(ptr, old_size);
        nonnull_aligned()
    } else if old_size == 0 {
        // The old "allocation" was just an aligned pointer, so we can just
        // allocate new memory.
        allocate_aligned(new_size)
    } else {
        check_size(new_size);
        let old_layout = layout_aligned(old_size);
        let new_ptr = std::alloc::realloc(ptr.as_ptr(), old_layout, new_size);
        // The allocator indicates an error by returning a null pointer.
        if new_ptr.is_null() {
            // Abort on error.
            std::alloc::handle_alloc_error(layout_aligned(new_size))
        } else {
            NonNull::new_unchecked(new_ptr)
        }
    }
}

pub unsafe fn memcpy(dst: *mut u8, src: *const u8, len: usize) {
    std::ptr::copy_nonoverlapping(src, dst, len)
}

extern "C" {
    pub fn memcmp(p1: *const u8, p2: *const u8, len: usize) -> i32;
}

/// Check if the pointer `p` is aligned to offset `a`.
pub fn is_aligned<T>(p: *const T, a: usize) -> bool {
    let a_minus_one = a.wrapping_sub(1);
    let pmoda = p as usize & a_minus_one;
    pmoda == 0
}

pub fn is_ptr_aligned<T>(p: *const T) -> bool {
    let alignment = align_of::<T>();
    is_aligned(p, alignment)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate() {
        for _ in 0..10 {
            let p = allocate_aligned(1024).as_ptr();
            // make sure this is 64-byte aligned
            assert_eq!(0, (p as usize) % 64);
        }
    }

    #[test]
    fn test_is_aligned() {
        // allocate memory aligned to 64-byte
        let mut ptr = allocate_aligned(10).as_ptr();
        assert_eq!(true, is_aligned::<u8>(ptr, 1));
        assert_eq!(true, is_aligned::<u8>(ptr, 2));
        assert_eq!(true, is_aligned::<u8>(ptr, 4));

        // now make the memory aligned to 63-byte
        ptr = unsafe { ptr.offset(1) };
        assert_eq!(true, is_aligned::<u8>(ptr, 1));
        assert_eq!(false, is_aligned::<u8>(ptr, 2));
        assert_eq!(false, is_aligned::<u8>(ptr, 4));
    }
}
