// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/accelerators/dx11_alloc_resource.hpp"
#include "streaming/onevpl/accelerators/utils/shared_lock.hpp"
#include "logger.hpp"

#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

LockAdapter::LockAdapter(mfxFrameAllocator origin_allocator) :
    lockable_allocator(origin_allocator),
    impl() {
    GAPI_DbgAssert((lockable_allocator.Lock && lockable_allocator.Unlock) &&
                   "Cannot create LockAdapter for empty origin allocator");

    // abandon unusable c-allocator interfaces
    // because LockAdapter requires Lock & Unlock only
    lockable_allocator.Alloc = nullptr;
    lockable_allocator.Free = nullptr;
    lockable_allocator.pthis = nullptr;
}

size_t LockAdapter::read_lock(mfxMemId mid, mfxFrameData &data) {
    size_t prev_lock_count = 0;
    if (impl) {
        prev_lock_count = impl->shared_lock();
    }

    // dispatch to VPL allocator using READ access mode
    mfxStatus sts = MFX_ERR_LOCK_MEMORY;
    try {
        sts = lockable_allocator.Lock(nullptr, mid, &data);
    } catch(...) {
    }

    // adapter will throw error if VPL frame allocator fails
    if (sts != MFX_ERR_NONE) {
        impl->unlock_shared();
        GAPI_Error("Cannot lock frame on READ using VPL allocator");
    }

    return prev_lock_count;
}

size_t LockAdapter::unlock_read(mfxMemId mid, mfxFrameData &data) {
    GAPI_DbgAssert(!impl || !is_write_acquired() &&
                   "Reject `unlock_read` in `write_lock` state");
    lockable_allocator.Unlock(nullptr, mid, &data);
    return impl ? impl->unlock_shared() : 0;
}

void LockAdapter::write_lock(mfxMemId mid, mfxFrameData &data) {
    if (impl) {
        // TODO consider using `try_lock` in loop with limited iteration count
        // to prevent dead-lock with WARN at least notification
        impl->lock();
    }

    // dispatch to VPL allocator using READ access mode
    mfxStatus sts = MFX_ERR_LOCK_MEMORY;
    try {
        sts = lockable_allocator.Lock(nullptr, mid, &data);
    } catch(...) {
    }

    // adapter will throw error if VPL frame allocator fails
    if (sts != MFX_ERR_NONE) {
        impl->unlock();
        GAPI_Error("Cannot lock frame on WRITE using VPL allocator");
    }
}

bool LockAdapter::is_write_acquired() {
    if(!impl) return true;
    return impl->owns();
}

void LockAdapter::unlock_write(mfxMemId mid, mfxFrameData &data) {
    GAPI_DbgAssert(is_write_acquired() &&
                   "Reject `unlock_write` for unlocked state");
    lockable_allocator.Unlock(nullptr, mid, &data);
    if (impl) {
        impl->unlock();
    }
}

SharedLock* LockAdapter::set_adaptee(SharedLock* new_impl) {
    SharedLock* old_impl = impl;
    GAPI_LOG_DEBUG(nullptr, "this: " << this <<
                            ", old: " << old_impl << ", new: " << new_impl);
    GAPI_DbgAssert(old_impl == nullptr || new_impl == nullptr && "Must not be previous impl");
    impl = new_impl;
    return old_impl;
}

SharedLock* LockAdapter::get_adaptee() {
    return impl;
}

NativeHandleAdapter::NativeHandleAdapter(mfxFrameAllocator origin_allocator) :
    native_handle_getter(origin_allocator) {
    GAPI_DbgAssert(native_handle_getter.GetHDL &&
                   "Cannot create NativeHandleAdapter for empty origin allocator");

    // abandon unusable c-allocator interfaces
    // because NativeHandleAdapter requires `GetHDL` only
    native_handle_getter.Alloc = nullptr;
    native_handle_getter.Free = nullptr;
    native_handle_getter.Lock = nullptr;
    native_handle_getter.Unlock = nullptr;
    native_handle_getter.pthis = nullptr;
}

void NativeHandleAdapter::get_handle(mfxMemId mid, mfxHDL& out) {
    if (native_handle_getter.GetHDL(nullptr, mid, &out) != MFX_ERR_NONE) {
        GAPI_Assert(nullptr && "Cannot get native handle for resourse by mid");
    }
}

DX11AllocationItem::DX11AllocationItem(std::weak_ptr<DX11AllocationRecord> parent,
                                       ID3D11DeviceContext* origin_ctx,
                                       mfxFrameAllocator origin_allocator,
                                       ComSharedPtrGuard<ID3D11Texture2D> tex_ptr,
                                       subresource_id_t subtex_id,
                                       ComPtrGuard<ID3D11Texture2D>&& staging_tex_ptr) :
    LockAdapter(origin_allocator),
    NativeHandleAdapter(origin_allocator),
    shared_device_context(origin_ctx),
    texture_ptr(tex_ptr),
    subresource_id(subtex_id),
    staging_texture_ptr(std::move(staging_tex_ptr)),
    observer(parent) {
    GAPI_DbgAssert(texture_ptr &&
                   "Cannot create DX11AllocationItem for empty texture");
    GAPI_DbgAssert(staging_texture_ptr &&
                   "Cannot create DX11AllocationItem for empty staging texture");
    GAPI_DbgAssert(observer.lock() &&
                   "Cannot create DX11AllocationItem for empty parent");

    shared_device_context->AddRef();
}

DX11AllocationItem::~DX11AllocationItem() {
    release();
    observer.reset();
    if (shared_device_context) {
        shared_device_context->Release();
    }
}

void DX11AllocationItem::release() {
    auto parent = observer.lock();
    GAPI_LOG_DEBUG(nullptr, "texture: " << texture_ptr <<
                            ", subresource id: " << subresource_id <<
                            ", parent: " << parent.get());
    cv::util::suppress_unused_warning(parent);
}

ID3D11Texture2D* DX11AllocationItem::get_texture_ptr() {
    return texture_ptr.get();
}

ID3D11Texture2D* DX11AllocationItem::get_staging_texture_ptr() {
    return staging_texture_ptr.get();
}

DX11AllocationItem::subresource_id_t DX11AllocationItem::get_subresource() const {
    return subresource_id;
}

ID3D11DeviceContext* DX11AllocationItem::get_device_ctx_ptr() {
    return shared_device_context;//shared_device_context.get();
}

void DX11AllocationItem::on_first_in_impl(mfxFrameData *ptr) {
    D3D11_MAP mapType = D3D11_MAP_READ;
    UINT mapFlags = D3D11_MAP_FLAG_DO_NOT_WAIT;

    GAPI_LOG_DEBUG(nullptr, "texture: " << get_texture_ptr() <<
                            ", subresorce: " << get_subresource());
    shared_device_context->CopySubresourceRegion(get_staging_texture_ptr(), 0,
                                                 0, 0, 0,
                                                 get_texture_ptr(),
                                                 get_subresource(),
                                                 nullptr);
    HRESULT err = S_OK;
    D3D11_MAPPED_SUBRESOURCE lockedRect {};
    do {
        err = shared_device_context->Map(get_staging_texture_ptr(), 0, mapType, mapFlags, &lockedRect);
        if (S_OK != err && DXGI_ERROR_WAS_STILL_DRAWING != err) {
            GAPI_LOG_WARNING(nullptr, "Cannot Map staging texture in device context, error: " << std::to_string(HRESULT_CODE(err)));
            GAPI_Error("Cannot Map staging texture in device context");
        }
    } while (DXGI_ERROR_WAS_STILL_DRAWING == err);

    if (FAILED(err)) {
        GAPI_LOG_WARNING(nullptr, "Cannot lock frame");
        GAPI_Error("Cannot lock frame");
        return ;
    }

    D3D11_TEXTURE2D_DESC desc {};
    get_texture_ptr()->GetDesc(&desc);
    switch (desc.Format) {
        case DXGI_FORMAT_NV12:
            ptr->Pitch = (mfxU16)lockedRect.RowPitch;
            ptr->Y     = (mfxU8 *)lockedRect.pData;
            ptr->UV     = (mfxU8 *)lockedRect.pData + desc.Height * lockedRect.RowPitch;

            GAPI_Assert(ptr->Y && ptr->UV && "DXGI_FORMAT_NV12 locked frame data is nullptr");
            break;
        default:
            GAPI_LOG_WARNING(nullptr, "Unknown DXGI format: " << desc.Format);
            return;
    }
}

void DX11AllocationItem::on_last_out_impl(mfxFrameData *ptr) {
    shared_device_context->Unmap(get_staging_texture_ptr(), 0);
    if (ptr) {
        ptr->Pitch = 0;
        ptr->U = ptr->V = ptr->Y = 0;
        ptr->A = ptr->R = ptr->G = ptr->B = 0;
    }
}

mfxStatus DX11AllocationItem::acquire_access(mfxFrameData *ptr) {
    if (is_write_acquired()) {
        return exclusive_access_acquire_unsafe(ptr);
    }
    return shared_access_acquire_unsafe(ptr);
}

mfxStatus DX11AllocationItem::release_access(mfxFrameData *ptr) {
    if (is_write_acquired()) {
        return exclusive_access_release_unsafe(ptr);
    }
    return shared_access_release_unsafe(ptr);
}

mfxStatus DX11AllocationItem::shared_access_acquire_unsafe(mfxFrameData *ptr) {
    GAPI_LOG_DEBUG(nullptr, "acquire READ lock: " << this <<
                            ", texture: " << get_texture_ptr() <<
                            ", sub id: " << get_subresource());
    // shared access requires elastic barrier
    // first-in visited thread uses resource mapping on host memory
    // subsequent threads reuses mapped memory
    //
    // exclusive access is prohibited while any one shared access has been obtained
    visit_in(ptr);

    if (!(ptr->Y && (ptr->UV || (ptr->U && ptr->V)))) {
        GAPI_LOG_WARNING(nullptr, "No any data obtained: " << this);
        GAPI_DbgAssert(false && "shared access must provide data");
        return MFX_ERR_LOCK_MEMORY;
    }
    GAPI_LOG_DEBUG(nullptr, "READ access granted: " << this);
    return MFX_ERR_NONE;
}

mfxStatus DX11AllocationItem::shared_access_release_unsafe(mfxFrameData *ptr) {
    GAPI_LOG_DEBUG(nullptr, "releasing READ lock: " << this <<
                            ", texture: " << get_texture_ptr() <<
                            ", sub id: " << get_subresource());
    // releasing shared access requires elastic barrier
    // last-out thread must make memory unmapping then and only then no more
    // read access is coming. If another read-access goes into critical section
    // (or waiting for acees) we must drop off unmapping procedure
    visit_out(ptr);

    GAPI_LOG_DEBUG(nullptr, "access on READ released: " << this);
    return MFX_ERR_NONE;
}

mfxStatus DX11AllocationItem::exclusive_access_acquire_unsafe(mfxFrameData *ptr) {
    GAPI_LOG_DEBUG(nullptr, "acquire WRITE lock: " << this <<
                            ", texture: " << get_texture_ptr() <<
                            ", sub id: " << get_subresource());
    D3D11_MAP mapType = D3D11_MAP_WRITE;
    UINT mapFlags = D3D11_MAP_FLAG_DO_NOT_WAIT;

    HRESULT err = S_OK;
    D3D11_MAPPED_SUBRESOURCE lockedRect {};
    do {
        err = get_device_ctx_ptr()->Map(get_staging_texture_ptr(), 0, mapType, mapFlags, &lockedRect);
        if (S_OK != err && DXGI_ERROR_WAS_STILL_DRAWING != err) {
            GAPI_LOG_WARNING(nullptr, "Cannot Map staging texture in device context, error: " << std::to_string(HRESULT_CODE(err)));
            return MFX_ERR_LOCK_MEMORY;
        }
    } while (DXGI_ERROR_WAS_STILL_DRAWING == err);

    if (FAILED(err)) {
        GAPI_LOG_WARNING(nullptr, "Cannot lock frame");
        return MFX_ERR_LOCK_MEMORY;
    }

    D3D11_TEXTURE2D_DESC desc {};
    get_texture_ptr()->GetDesc(&desc);
    switch (desc.Format) {
        case DXGI_FORMAT_NV12:
            ptr->Pitch = (mfxU16)lockedRect.RowPitch;
            ptr->Y = (mfxU8 *)lockedRect.pData;
            ptr->UV = (mfxU8 *)lockedRect.pData + desc.Height * lockedRect.RowPitch;
            if (!ptr->Y || !ptr->UV) {
                GAPI_LOG_WARNING(nullptr, "DXGI_FORMAT_NV12 locked frame data is nullptr");
                return MFX_ERR_LOCK_MEMORY;
            }
            break;
        default:
            GAPI_LOG_WARNING(nullptr, "Unknown DXGI format: " << desc.Format);
            return MFX_ERR_LOCK_MEMORY;
    }

    GAPI_LOG_DEBUG(nullptr, "WRITE access granted: " << this);
    return MFX_ERR_NONE;
}

mfxStatus DX11AllocationItem::exclusive_access_release_unsafe(mfxFrameData *ptr) {
    GAPI_LOG_DEBUG(nullptr, "releasing WRITE lock: " << this <<
                            ", texture: " << get_texture_ptr() <<
                            ", sub id: " << get_subresource());

    get_device_ctx_ptr()->Unmap(get_staging_texture_ptr(), 0);

    get_device_ctx_ptr()->CopySubresourceRegion(get_texture_ptr(),
                                            get_subresource(),
                                            0, 0, 0,
                                            get_staging_texture_ptr(), 0,
                                            nullptr);

    if (ptr) {
        ptr->Pitch = 0;
        ptr->U = ptr->V = ptr->Y = 0;
        ptr->A = ptr->R = ptr->G = ptr->B = 0;
    }
    GAPI_LOG_DEBUG(nullptr, "access on WRITE released: " << this);
    return MFX_ERR_NONE;
}

DX11AllocationRecord::DX11AllocationRecord() = default;

DX11AllocationRecord::~DX11AllocationRecord() {
    GAPI_LOG_DEBUG(nullptr, "record: " << this <<
                            ", subresources count: " << resources.size());

    for (AllocationId id : resources) {
        delete id;
    }
    resources.clear();

    GAPI_LOG_DEBUG(nullptr, "release final referenced texture: " << texture_ptr.get());
}

void DX11AllocationRecord::init(unsigned int items, ID3D11DeviceContext* origin_ctx,
                                mfxFrameAllocator origin_allocator,
                                std::vector<ComPtrGuard<ID3D11Texture2D>> &&textures,
                                std::vector<ComPtrGuard<ID3D11Texture2D>> &&staging_textures) {

    GAPI_DbgAssert(items != 0 && "Cannot create DX11AllocationRecord with empty items");
    GAPI_DbgAssert(items == staging_textures.size() && "Allocation items count and staging size are not equal");
    GAPI_DbgAssert(textures.size() != 1 ? items == textures.size() : true && "Allocation items count and staging size are not equal");
    GAPI_DbgAssert(origin_ctx &&
                   "Cannot create DX11AllocationItem for empty origin_ctx");
    auto shared_allocator_copy = origin_allocator;
    GAPI_DbgAssert((shared_allocator_copy.Lock && shared_allocator_copy.Unlock) &&
                   "Cannot create DX11AllocationItem for empty origin allocator");

    // abandon unusable c-allocator interfaces
    shared_allocator_copy.Alloc = nullptr;
    shared_allocator_copy.Free = nullptr;
    shared_allocator_copy.pthis = nullptr;


    GAPI_LOG_DEBUG(nullptr, "subresources count: " << items);
    resources.reserve(items);

    if (textures.size() == 1) {
        texture_ptr = createCOMSharedPtrGuard(std::move(textures[0]));
    }
    for(unsigned int i = 0; i < items; i++) {
        if (textures.size() == 1) {
            GAPI_LOG_DEBUG(nullptr, "subresources: [" << i <<", " << items << "], ID3D11Texture2D: " << texture_ptr.get());
            resources.emplace_back(new DX11AllocationItem(get_ptr(), origin_ctx, shared_allocator_copy,
                                                          texture_ptr, i, std::move(staging_textures[i])));
        } else {
            GAPI_LOG_DEBUG(nullptr, "subresources: [" << i <<", " << items << "], ID3D11Texture2D: " << textures[i].get());
            resources.emplace_back(new DX11AllocationItem(get_ptr(), origin_ctx, shared_allocator_copy,
                                                          std::move(textures[i]), 0, std::move(staging_textures[i])));
        }
    }
}

DX11AllocationRecord::Ptr DX11AllocationRecord::get_ptr() {
    return shared_from_this();
}

DX11AllocationRecord::AllocationId* DX11AllocationRecord::data() {
    return resources.data();
}

size_t DX11AllocationRecord::size() const {
    return resources.size();
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_D3D11
#endif // HAVE_DIRECTX
#endif // HAVE_ONEVPL
