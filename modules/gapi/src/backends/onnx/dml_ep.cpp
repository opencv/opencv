// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation

#include "backends/onnx/dml_ep.hpp"
#include "logger.hpp"

#ifdef HAVE_ONNX
#include <onnxruntime_cxx_api.h>

#ifdef HAVE_ONNX_DML
#include "../providers/dml/dml_provider_factory.h"

// FIXME It must be different #ifdef
// E.g HAVE_DXCORE && HAVE_DIRECTX12 && HAVE_DIRECTML
#ifdef HAVE_DXCORE

// FIXME: Fix warning
#define WINVER 0x0A00
#define _WIN32_WINNT 0x0A00

#include <initguid.h>

#include <d3d11.h>
#include <dxgi1_2.h>
#include <dxgi1_4.h>
#include <dxgi.h>
#include <dxcore.h>
#include <dxcore_interface.h>
#include <d3d12.h>
#include <directml.h>

#pragma comment (lib, "d3d11.lib")
#pragma comment (lib, "d3d12.lib")
#pragma comment (lib, "dxgi.lib")
#pragma comment (lib, "dxcore.lib")
#pragma comment (lib, "directml.lib")

#endif  // HAVE_DXCORE

static void addDMLExecutionProviderWithAdapterName(Ort::SessionOptions *session_options,
                                                   const std::string &adapter_name);

void cv::gimpl::onnx::addDMLExecutionProvider(Ort::SessionOptions *session_options,
                                              const cv::gapi::onnx::ep::DirectML &dml_ep) {
    namespace ep = cv::gapi::onnx::ep;
    switch (dml_ep.ddesc.index()) {
        case ep::DirectML::DeviceDesc::index_of<int>(): {
            const int device_id = cv::util::get<int>(dml_ep.ddesc);
            try {
                OrtSessionOptionsAppendExecutionProvider_DML(*session_options, device_id);
            } catch (const std::exception &e) {
                std::stringstream ss;
                ss << "ONNX Backend: Failed to enable DirectML"
                   << " Execution Provider: " << e.what();
                cv::util::throw_error(std::runtime_error(ss.str()));
            }
            break;
        }
        case ep::DirectML::DeviceDesc::index_of<std::string>(): {
            const std::string adapter_name = cv::util::get<std::string>(dml_ep.ddesc);
            addDMLExecutionProviderWithAdapterName(session_options, adapter_name);
            break;
        }
        default:
            GAPI_Assert(false && "Invalid DirectML device description");
    }
}

#ifdef HAVE_DXCORE

#define THROW_IF_FAILED(hr)                                   \
{                                                             \
    if ((hr) != S_OK)                                         \
        throw std::logic_error("Failed with status != S_OK"); \
}

struct AdapterDesc {
    IDXCoreAdapter* ptr;
    char description[256];
};

static std::vector<AdapterDesc> getAvailableAdapters() {
        std::vector<AdapterDesc> all_adapters;

        IDXCoreAdapterFactory* factory;
        GAPI_LOG_DEBUG(nullptr, "Create IDXCoreAdapterFactory");
        THROW_IF_FAILED(DXCoreCreateAdapterFactory(__uuidof(IDXCoreAdapterFactory), (void**)&factory));
        GAPI_LOG_DEBUG(nullptr, "Create IDXCoreAdapterFactory - successful");

        IDXCoreAdapterList* adapter_list;
        const GUID dxGUIDs[] = { DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE };
        GAPI_LOG_DEBUG(nullptr, "CreateAdapterList");
        THROW_IF_FAILED(factory->CreateAdapterList(ARRAYSIZE(dxGUIDs), dxGUIDs, __uuidof(IDXCoreAdapterList), (void**)&adapter_list));
        GAPI_LOG_DEBUG(nullptr, "CreateAdapterList - successful");

        for (UINT i = 0; i < adapter_list->GetAdapterCount(); i++)
        {
            IDXCoreAdapter* curr_adapter;
            GAPI_LOG_DEBUG(nullptr, "GetAdapter");
            THROW_IF_FAILED(adapter_list->GetAdapter(i, __uuidof(IDXCoreAdapter), (void**)&curr_adapter));
            GAPI_LOG_DEBUG(nullptr, "GetAdapter - successful");

            bool is_hardware = false;
            curr_adapter->GetProperty(DXCoreAdapterProperty::IsHardware, &is_hardware);
            // NB: Filter out if not hardware adapter.
            if (!is_hardware) {
                continue;
            }

            AdapterDesc adapter_desc;
            adapter_desc.ptr = curr_adapter;

            size_t desc_size;
            curr_adapter->GetPropertySize(DXCoreAdapterProperty::DriverDescription, &desc_size);
            curr_adapter->GetProperty(DXCoreAdapterProperty::DriverDescription, desc_size, &adapter_desc.description);
            all_adapters.push_back(std::move(adapter_desc));
        }
        return all_adapters;
};

struct DMLDeviceInfo {
    IDMLDevice* device;
    ID3D12CommandQueue* cmd_queue;
};

static DMLDeviceInfo createDMLInfo(IDXCoreAdapter* adapter) {
    IUnknown* pAdapter = adapter;
    D3D_FEATURE_LEVEL d3dFeatureLevel = D3D_FEATURE_LEVEL_1_0_CORE;
    if (adapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS))
    {
        GAPI_LOG_INFO(nullptr, "DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS is supported");
        d3dFeatureLevel = D3D_FEATURE_LEVEL::D3D_FEATURE_LEVEL_11_0;

        IDXGIFactory4* dxgiFactory4;
        GAPI_LOG_DEBUG(nullptr, "CreateDXGIFactory2");
        THROW_IF_FAILED(CreateDXGIFactory2(0, __uuidof(IDXGIFactory4), (void**)&dxgiFactory4));
        GAPI_LOG_DEBUG(nullptr, "CreateDXGIFactory2 - successful");
        // If DXGI factory creation was successful then get the IDXGIAdapter from the LUID
        // acquired from the selectedAdapter
        LUID adapterLuid;
        IDXGIAdapter* spDxgiAdapter;

        GAPI_LOG_DEBUG(nullptr, "Get DXCoreAdapterProperty::InstanceLuid property");
        THROW_IF_FAILED(adapter->GetProperty(DXCoreAdapterProperty::InstanceLuid, &adapterLuid));
        GAPI_LOG_DEBUG(nullptr, "Get DXCoreAdapterProperty::InstanceLuid property - successful");

        GAPI_LOG_DEBUG(nullptr, "Get IDXGIAdapter by luid");
        THROW_IF_FAILED(dxgiFactory4->EnumAdapterByLuid(adapterLuid, __uuidof(IDXGIAdapter), (void**)&spDxgiAdapter));
        GAPI_LOG_DEBUG(nullptr, "Get IDXGIAdapter by luid - successful");

        pAdapter = spDxgiAdapter;
    } else {
        GAPI_LOG_INFO(nullptr, "DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS isn't supported");
    }

    ID3D12Device* d3d12Device;
    GAPI_LOG_DEBUG(nullptr, "Create D3D12Device");
    THROW_IF_FAILED(D3D12CreateDevice(pAdapter, d3dFeatureLevel, __uuidof(ID3D12Device), (void**)&d3d12Device));
    GAPI_LOG_DEBUG(nullptr, "Create D3D12Device - successful");

    D3D12_COMMAND_LIST_TYPE commandQueueType = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    ID3D12CommandQueue* d3d12CommandQueue;
    D3D12_COMMAND_QUEUE_DESC commandQueueDesc = {};
    commandQueueDesc.Type = commandQueueType;
    GAPI_LOG_DEBUG(nullptr, "Create D3D12CommandQueue");
    THROW_IF_FAILED(d3d12Device->CreateCommandQueue(&commandQueueDesc, __uuidof(ID3D12CommandQueue), (void**)&d3d12CommandQueue));
    GAPI_LOG_DEBUG(nullptr, "Create D3D12CommandQueue - successful");

    IDMLDevice* dmlDevice;
    GAPI_LOG_DEBUG(nullptr, "Create DirectML device");
    THROW_IF_FAILED(DMLCreateDevice(d3d12Device, DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&dmlDevice)));
    GAPI_LOG_DEBUG(nullptr, "Create DirectML device - successful");

    return {dmlDevice, d3d12CommandQueue};
};

static void addDMLExecutionProviderWithAdapterName(Ort::SessionOptions *session_options,
                                                   const std::string &adapter_name) {
    const auto all_adapters = getAvailableAdapters();

    std::vector<AdapterDesc> selected_adapters;
    std::stringstream log_msg;
    for (const auto& adapter : all_adapters) {
        log_msg << adapter.description << std::endl;
        if (std::strstr(adapter.description, adapter_name.c_str())) {
            selected_adapters.emplace_back(adapter);
        }
    }
    GAPI_LOG_INFO(NULL, "\nAvailable DirectML adapters:\n" << log_msg.str());

    if (selected_adapters.empty()) {
        std::stringstream error_msg;
        error_msg << "ONNX Backend: No DirectML adapters found match to \"" << adapter_name << "\"";
        cv::util::throw_error(std::runtime_error(error_msg.str()));
    } else if (selected_adapters.size() > 1) {
        std::stringstream error_msg;
        error_msg << "ONNX Backend: More than one adapter matches to \"" << adapter_name << "\":\n";
        for (const auto &selected_adapter : selected_adapters) {
            error_msg << selected_adapter.description << "\n";
        }
        cv::util::throw_error(std::runtime_error(error_msg.str()));
    }

    GAPI_LOG_INFO(NULL, "Selected device: " << selected_adapters.front().description);
    auto dml = createDMLInfo(selected_adapters.front().ptr);
    try {
        OrtSessionOptionsAppendExecutionProviderEx_DML(*session_options, dml.device, dml.cmd_queue);
    } catch (const std::exception &e) {
        std::stringstream ss;
        ss << "ONNX Backend: Failed to enable DirectML"
           << " Execution Provider: " << e.what();
        cv::util::throw_error(std::runtime_error(ss.str()));
    }
}

#else  // HAVE_DXCORE

static void addDMLExecutionProviderWithAdapterName(Ort::SessionOptions*, const std::string&) {
    std::stringstream ss;
    ss << "ONNX Backend: Failed to add DirectML Execution Provider with adapter name."
       << " DirectML support is required.";
    cv::util::throw_error(std::runtime_error(ss.str()));
}

#endif  // HAVE_DXCORE
#else  // HAVE_ONNX_DML

void cv::gimpl::onnx::addDMLExecutionProvider(Ort::SessionOptions*,
                                              const cv::gapi::onnx::ep::DirectML&) {
     util::throw_error(std::runtime_error("G-API has been compiled with ONNXRT"
                                          " without DirectML support"));
}

#endif  // HAVE_ONNX_DML
#endif  // HAVE_ONNX
