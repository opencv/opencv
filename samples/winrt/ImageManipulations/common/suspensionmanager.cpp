//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

//
// SuspensionManager.cpp
// Implementation of the SuspensionManager class
//

#include "pch.h"
#include "SuspensionManager.h"

#include <collection.h>
#include <algorithm>

using namespace SDKSample::Common;

using namespace Concurrency;
using namespace Platform;
using namespace Platform::Collections;
using namespace Windows::Foundation;
using namespace Windows::Foundation::Collections;
using namespace Windows::Storage;
using namespace Windows::Storage::FileProperties;
using namespace Windows::Storage::Streams;
using namespace Windows::UI::Xaml;
using namespace Windows::UI::Xaml::Controls;
using namespace Windows::UI::Xaml::Interop;

namespace
{
    Map<String^, Object^>^ _sessionState = ref new Map<String^, Object^>();
    String^ sessionStateFilename = "_sessionState.dat";

    // Forward declarations for object object read / write support
    void WriteObject(Windows::Storage::Streams::DataWriter^ writer, Platform::Object^ object);
    Platform::Object^ ReadObject(Windows::Storage::Streams::DataReader^ reader);
}

/// <summary>
/// Provides access to global session state for the current session.  This state is serialized by
/// <see cref="SaveAsync"/> and restored by <see cref="RestoreAsync"/> which require values to be
/// one of the following: boxed values including integers, floating-point singles and doubles,
/// wide characters, boolean, Strings and Guids, or Map<String^, Object^> where map values are
/// subject to the same constraints.  Session state should be as compact as possible.
/// </summary>
IMap<String^, Object^>^ SuspensionManager::SessionState::get(void)
{
    return _sessionState;
}

/// <summary>
/// Wrap a WeakReference as a reference object for use in a collection.
/// </summary>
private ref class WeakFrame sealed
{
private:
    WeakReference _frameReference;

internal:
    WeakFrame(Frame^ frame) { _frameReference = frame; }
    property Frame^ ResolvedFrame
    {
        Frame^ get(void) { return _frameReference.Resolve<Frame>(); }
    };
};

namespace
{
    std::vector<WeakFrame^> _registeredFrames;
    DependencyProperty^ FrameSessionStateKeyProperty =
        DependencyProperty::RegisterAttached("_FrameSessionStateKeyProperty",
        TypeName(String::typeid), TypeName(SuspensionManager::typeid), nullptr);
    DependencyProperty^ FrameSessionStateProperty =
        DependencyProperty::RegisterAttached("_FrameSessionStateProperty",
        TypeName(IMap<String^, Object^>::typeid), TypeName(SuspensionManager::typeid), nullptr);
}

/// <summary>
/// Registers a <see cref="Frame"/> instance to allow its navigation history to be saved to
/// and restored from <see cref="SessionState"/>.  Frames should be registered once
/// immediately after creation if they will participate in session state management.  Upon
/// registration if state has already been restored for the specified key
/// the navigation history will immediately be restored.  Subsequent invocations of
/// <see cref="RestoreAsync(String)"/> will also restore navigation history.
/// </summary>
/// <param name="frame">An instance whose navigation history should be managed by
/// <see cref="SuspensionManager"/></param>
/// <param name="sessionStateKey">A unique key into <see cref="SessionState"/> used to
/// store navigation-related information.</param>
void SuspensionManager::RegisterFrame(Frame^ frame, String^ sessionStateKey)
{
    if (frame->GetValue(FrameSessionStateKeyProperty) != nullptr)
    {
        throw ref new FailureException("Frames can only be registered to one session state key");
    }

    if (frame->GetValue(FrameSessionStateProperty) != nullptr)
    {
        throw ref new FailureException("Frames must be either be registered before accessing frame session state, or not registered at all");
    }

    // Use a dependency property to associate the session key with a frame, and keep a list of frames whose
    // navigation state should be managed
    frame->SetValue(FrameSessionStateKeyProperty, sessionStateKey);
    _registeredFrames.insert(_registeredFrames.begin(), ref new WeakFrame(frame));

    // Check to see if navigation state can be restored
    RestoreFrameNavigationState(frame);
}

/// <summary>
/// Disassociates a <see cref="Frame"/> previously registered by <see cref="RegisterFrame"/>
/// from <see cref="SessionState"/>.  Any navigation state previously captured will be
/// removed.
/// </summary>
/// <param name="frame">An instance whose navigation history should no longer be
/// managed.</param>
void SuspensionManager::UnregisterFrame(Frame^ frame)
{
    // Remove session state and remove the frame from the list of frames whose navigation
    // state will be saved (along with any weak references that are no longer reachable)
    auto key = safe_cast<String^>(frame->GetValue(FrameSessionStateKeyProperty));
    if (SessionState->HasKey(key)) SessionState->Remove(key);
    _registeredFrames.erase(
        std::remove_if(_registeredFrames.begin(), _registeredFrames.end(), [=](WeakFrame^& e)
        {
            auto testFrame = e->ResolvedFrame;
            return testFrame == nullptr || testFrame == frame;
        }),
        _registeredFrames.end()
    );
}

/// <summary>
/// Provides storage for session state associated with the specified <see cref="Frame"/>.
/// Frames that have been previously registered with <see cref="RegisterFrame"/> have
/// their session state saved and restored automatically as a part of the global
/// <see cref="SessionState"/>.  Frames that are not registered have transient state
/// that can still be useful when restoring pages that have been discarded from the
/// navigation cache.
/// </summary>
/// <remarks>Apps may choose to rely on <see cref="LayoutAwarePage"/> to manage
/// page-specific state instead of working with frame session state directly.</remarks>
/// <param name="frame">The instance for which session state is desired.</param>
/// <returns>A collection of state subject to the same serialization mechanism as
/// <see cref="SessionState"/>.</returns>
IMap<String^, Object^>^ SuspensionManager::SessionStateForFrame(Frame^ frame)
{
    auto frameState = safe_cast<IMap<String^, Object^>^>(frame->GetValue(FrameSessionStateProperty));

    if (frameState == nullptr)
    {
        auto frameSessionKey = safe_cast<String^>(frame->GetValue(FrameSessionStateKeyProperty));
        if (frameSessionKey != nullptr)
        {
            // Registered frames reflect the corresponding session state
            if (!_sessionState->HasKey(frameSessionKey))
            {
                _sessionState->Insert(frameSessionKey, ref new Map<String^, Object^>());
            }
            frameState = safe_cast<IMap<String^, Object^>^>(_sessionState->Lookup(frameSessionKey));
        }
        else
        {
            // Frames that aren't registered have transient state
            frameState = ref new Map<String^, Object^>();
        }
        frame->SetValue(FrameSessionStateProperty, frameState);
    }
    return frameState;
}

void SuspensionManager::RestoreFrameNavigationState(Frame^ frame)
{
    auto frameState = SessionStateForFrame(frame);
    if (frameState->HasKey("Navigation"))
    {
        frame->SetNavigationState(safe_cast<String^>(frameState->Lookup("Navigation")));
    }
}

void SuspensionManager::SaveFrameNavigationState(Frame^ frame)
{
    auto frameState = SessionStateForFrame(frame);
    frameState->Insert("Navigation", frame->GetNavigationState());
}

/// <summary>
/// Save the current <see cref="SessionState"/>.  Any <see cref="Frame"/> instances
/// registered with <see cref="RegisterFrame"/> will also preserve their current
/// navigation stack, which in turn gives their active <see cref="Page"/> an opportunity
/// to save its state.
/// </summary>
/// <returns>An asynchronous task that reflects when session state has been saved.</returns>
task<void> SuspensionManager::SaveAsync(void)
{
    // Save the navigation state for all registered frames
    for (auto&& weakFrame : _registeredFrames)
    {
        auto frame = weakFrame->ResolvedFrame;
        if (frame != nullptr) SaveFrameNavigationState(frame);
    }

    // Serialize the session state synchronously to avoid asynchronous access to shared
    // state
    auto sessionData = ref new InMemoryRandomAccessStream();
    auto sessionDataWriter = ref new DataWriter(sessionData->GetOutputStreamAt(0));
    WriteObject(sessionDataWriter, _sessionState);

    // Once session state has been captured synchronously, begin the asynchronous process
    // of writing the result to disk
    return task<unsigned int>(sessionDataWriter->StoreAsync()).then([=](unsigned int)
    {
        return sessionDataWriter->FlushAsync();
    }).then([=](bool flushSucceeded)
    {
        (void)flushSucceeded; // Unused parameter
        return ApplicationData::Current->LocalFolder->CreateFileAsync(sessionStateFilename,
            CreationCollisionOption::ReplaceExisting);
    }).then([=](StorageFile^ createdFile)
    {
        return createdFile->OpenAsync(FileAccessMode::ReadWrite);
    }).then([=](IRandomAccessStream^ newStream)
    {
        return RandomAccessStream::CopyAndCloseAsync(
            sessionData->GetInputStreamAt(0), newStream->GetOutputStreamAt(0));
    }).then([=](UINT64 copiedBytes)
    {
        (void)copiedBytes; // Unused parameter
        return;
    });
}

/// <summary>
/// Restores previously saved <see cref="SessionState"/>.  Any <see cref="Frame"/> instances
/// registered with <see cref="RegisterFrame"/> will also restore their prior navigation
/// state, which in turn gives their active <see cref="Page"/> an opportunity restore its
/// state.
/// </summary>
/// <param name="version">A version identifier compared to the session state to prevent
/// incompatible versions of session state from reaching app code.  Saved state with a
/// different version will be ignored, resulting in an empty <see cref="SessionState"/>
/// dictionary.</param>
/// <returns>An asynchronous task that reflects when session state has been read.  The
/// content of <see cref="SessionState"/> should not be relied upon until this task
/// completes.</returns>
task<void> SuspensionManager::RestoreAsync(void)
{
    _sessionState->Clear();

    task<StorageFile^> getFileTask(ApplicationData::Current->LocalFolder->GetFileAsync(sessionStateFilename));
    return getFileTask.then([=](StorageFile^ stateFile)
    {
        task<BasicProperties^> getBasicPropertiesTask(stateFile->GetBasicPropertiesAsync());
        return getBasicPropertiesTask.then([=](BasicProperties^ stateFileProperties)
        {
            auto size = unsigned int(stateFileProperties->Size);
            if (size != stateFileProperties->Size) throw ref new FailureException("Session state larger than 4GB");
            task<IRandomAccessStreamWithContentType^> openReadTask(stateFile->OpenReadAsync());
            return openReadTask.then([=](IRandomAccessStreamWithContentType^ stateFileStream)
            {
                auto stateReader = ref new DataReader(stateFileStream);
                return task<unsigned int>(stateReader->LoadAsync(size)).then([=](unsigned int bytesRead)
                {
                    (void)bytesRead; // Unused parameter
                    // Deserialize the Session State
                    Object^ content = ReadObject(stateReader);
                    _sessionState = (Map<String^, Object^>^)content;

                    // Restore any registered frames to their saved state
                    for (auto&& weakFrame : _registeredFrames)
                    {
                        auto frame = weakFrame->ResolvedFrame;
                        if (frame != nullptr)
                        {
                            frame->ClearValue(FrameSessionStateProperty);
                            RestoreFrameNavigationState(frame);
                        }
                    }
                }, task_continuation_context::use_current());
            });
        });
    });
}

#pragma region Object serialization for a known set of types

namespace
{
    // Codes used for identifying serialized types
    enum StreamTypes {
        NullPtrType = 0,

        // Supported IPropertyValue types
        UInt8Type, UInt16Type, UInt32Type, UInt64Type, Int16Type, Int32Type, Int64Type,
        SingleType, DoubleType, BooleanType, Char16Type, GuidType, StringType,

        // Additional supported types
        StringToObjectMapType,

        // Marker values used to ensure stream integrity
        MapEndMarker
    };

    void WriteString(DataWriter^ writer, String^ string)
    {
        writer->WriteByte(StringType);
        writer->WriteUInt32(writer->MeasureString(string));
        writer->WriteString(string);
    }

    void WriteProperty(DataWriter^ writer, IPropertyValue^ propertyValue)
    {
        switch (propertyValue->Type)
        {
        case PropertyType::UInt8:
            writer->WriteByte(UInt8Type);
            writer->WriteByte(propertyValue->GetUInt8());
            return;
        case PropertyType::UInt16:
            writer->WriteByte(UInt16Type);
            writer->WriteUInt16(propertyValue->GetUInt16());
            return;
        case PropertyType::UInt32:
            writer->WriteByte(UInt32Type);
            writer->WriteUInt32(propertyValue->GetUInt32());
            return;
        case PropertyType::UInt64:
            writer->WriteByte(UInt64Type);
            writer->WriteUInt64(propertyValue->GetUInt64());
            return;
        case PropertyType::Int16:
            writer->WriteByte(Int16Type);
            writer->WriteUInt16(propertyValue->GetInt16());
            return;
        case PropertyType::Int32:
            writer->WriteByte(Int32Type);
            writer->WriteUInt32(propertyValue->GetInt32());
            return;
        case PropertyType::Int64:
            writer->WriteByte(Int64Type);
            writer->WriteUInt64(propertyValue->GetInt64());
            return;
        case PropertyType::Single:
            writer->WriteByte(SingleType);
            writer->WriteSingle(propertyValue->GetSingle());
            return;
        case PropertyType::Double:
            writer->WriteByte(DoubleType);
            writer->WriteDouble(propertyValue->GetDouble());
            return;
        case PropertyType::Boolean:
            writer->WriteByte(BooleanType);
            writer->WriteBoolean(propertyValue->GetBoolean());
            return;
        case PropertyType::Char16:
            writer->WriteByte(Char16Type);
            writer->WriteUInt16(propertyValue->GetChar16());
            return;
        case PropertyType::Guid:
            writer->WriteByte(GuidType);
            writer->WriteGuid(propertyValue->GetGuid());
            return;
        case PropertyType::String:
            WriteString(writer, propertyValue->GetString());
            return;
        default:
            throw ref new InvalidArgumentException("Unsupported property type");
        }
    }

    void WriteStringToObjectMap(DataWriter^ writer, IMap<String^, Object^>^ map)
    {
        writer->WriteByte(StringToObjectMapType);
        writer->WriteUInt32(map->Size);
        for (auto&& pair : map)
        {
            WriteObject(writer, pair->Key);
            WriteObject(writer, pair->Value);
        }
        writer->WriteByte(MapEndMarker);
    }

    void WriteObject(DataWriter^ writer, Object^ object)
    {
        if (object == nullptr)
        {
            writer->WriteByte(NullPtrType);
            return;
        }

        auto propertyObject = dynamic_cast<IPropertyValue^>(object);
        if (propertyObject != nullptr)
        {
            WriteProperty(writer, propertyObject);
            return;
        }

        auto mapObject = dynamic_cast<IMap<String^, Object^>^>(object);
        if (mapObject != nullptr)
        {
            WriteStringToObjectMap(writer, mapObject);
            return;
        }

        throw ref new InvalidArgumentException("Unsupported data type");
    }

    String^ ReadString(DataReader^ reader)
    {
        int length = reader->ReadUInt32();
        String^ string = reader->ReadString(length);
        return string;
    }

    IMap<String^, Object^>^ ReadStringToObjectMap(DataReader^ reader)
    {
        auto map = ref new Map<String^, Object^>();
        auto size = reader->ReadUInt32();
        for (unsigned int index = 0; index < size; index++)
        {
            auto key = safe_cast<String^>(ReadObject(reader));
            auto value = ReadObject(reader);
            map->Insert(key, value);
        }
        if (reader->ReadByte() != MapEndMarker)
        {
            throw ref new InvalidArgumentException("Invalid stream");
        }
        return map;
    }

    Object^ ReadObject(DataReader^ reader)
    {
        auto type = reader->ReadByte();
        switch (type)
        {
        case NullPtrType:
            return nullptr;
        case UInt8Type:
            return reader->ReadByte();
        case UInt16Type:
            return reader->ReadUInt16();
        case UInt32Type:
            return reader->ReadUInt32();
        case UInt64Type:
            return reader->ReadUInt64();
        case Int16Type:
            return reader->ReadInt16();
        case Int32Type:
            return reader->ReadInt32();
        case Int64Type:
            return reader->ReadInt64();
        case SingleType:
            return reader->ReadSingle();
        case DoubleType:
            return reader->ReadDouble();
        case BooleanType:
            return reader->ReadBoolean();
        case Char16Type:
            return (char16_t)reader->ReadUInt16();
        case GuidType:
            return reader->ReadGuid();
        case StringType:
            return ReadString(reader);
        case StringToObjectMapType:
            return ReadStringToObjectMap(reader);
        default:
            throw ref new InvalidArgumentException("Unsupported property type");
        }
    }
}

#pragma endregion
