#include <initguid.h>

#include <d3d11.h>
#include <dxgi1_2.h>
#include <dxgi1_4.h>
#include <dxgi.h>
#include <dxcore.h>
#include <dxcore_interface.h>
#include <d3d12.h>
#include <directml.h>

int main(int /*argc*/, char** /*argv*/)
{
    IDXCoreAdapterFactory* factory;
    DXCoreCreateAdapterFactory(__uuidof(IDXCoreAdapterFactory), (void**)&factory);

    IDXCoreAdapterList* adapterList;
    const GUID dxGUIDs[] = { DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE };
    factory->CreateAdapterList(ARRAYSIZE(dxGUIDs), dxGUIDs, __uuidof(IDXCoreAdapterList), (void**)&adapterList);

    IDXCoreAdapter* adapter;
    adapterList->GetAdapter(0u, __uuidof(IDXCoreAdapter), (void**)&adapter);

    D3D_FEATURE_LEVEL d3dFeatureLevel = D3D_FEATURE_LEVEL_1_0_CORE;
    ID3D12Device* d3d12Device = NULL;
    D3D12CreateDevice((IUnknown*)adapter, d3dFeatureLevel, __uuidof(ID3D11Device), (void**)&d3d12Device);

    D3D12_COMMAND_LIST_TYPE commandQueueType = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    ID3D12CommandQueue* cmdQueue;
    D3D12_COMMAND_QUEUE_DESC commandQueueDesc = {};
    commandQueueDesc.Type = commandQueueType;

    d3d12Device->CreateCommandQueue(&commandQueueDesc, __uuidof(ID3D12CommandQueue), (void**)&cmdQueue);
    IDMLDevice* dmlDevice;
    DMLCreateDevice(d3d12Device, DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&dmlDevice));

    return 0;
}