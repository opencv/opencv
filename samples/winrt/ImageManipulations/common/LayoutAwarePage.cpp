//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

#include "pch.h"
#include "LayoutAwarePage.h"
#include "SuspensionManager.h"

using namespace SDKSample::Common;

using namespace Platform;
using namespace Platform::Collections;
using namespace Windows::Foundation;
using namespace Windows::Foundation::Collections;
using namespace Windows::System;
using namespace Windows::UI::Core;
using namespace Windows::UI::ViewManagement;
using namespace Windows::UI::Xaml;
using namespace Windows::UI::Xaml::Controls;
using namespace Windows::UI::Xaml::Interop;
using namespace Windows::UI::Xaml::Navigation;

/// <summary>
/// Initializes a new instance of the <see cref="LayoutAwarePage"/> class.
/// </summary>
LayoutAwarePage::LayoutAwarePage()
{
    if (Windows::ApplicationModel::DesignMode::DesignModeEnabled)
    {
        return;
    }

    // Create an empty default view model
    DefaultViewModel = ref new Map<String^, Object^>(std::less<String^>());

    // When this page is part of the visual tree make two changes:
    // 1) Map application view state to visual state for the page
    // 2) Handle keyboard and mouse navigation requests
    Loaded += ref new RoutedEventHandler(this, &LayoutAwarePage::OnLoaded);

    // Undo the same changes when the page is no longer visible
    Unloaded += ref new RoutedEventHandler(this, &LayoutAwarePage::OnUnloaded);
}

static DependencyProperty^ _defaultViewModelProperty =
    DependencyProperty::Register("DefaultViewModel",
    TypeName(IObservableMap<String^, Object^>::typeid), TypeName(LayoutAwarePage::typeid), nullptr);

/// <summary>
/// Identifies the <see cref="DefaultViewModel"/> dependency property.
/// </summary>
DependencyProperty^ LayoutAwarePage::DefaultViewModelProperty::get()
{
    return _defaultViewModelProperty;
}

/// <summary>
/// Gets an implementation of <see cref="IObservableMap&lt;String, Object&gt;"/> designed to be
/// used as a trivial view model.
/// </summary>
IObservableMap<String^, Object^>^ LayoutAwarePage::DefaultViewModel::get()
{
    return safe_cast<IObservableMap<String^, Object^>^>(GetValue(DefaultViewModelProperty));
}

/// <summary>
/// Sets an implementation of <see cref="IObservableMap&lt;String, Object&gt;"/> designed to be
/// used as a trivial view model.
/// </summary>
void LayoutAwarePage::DefaultViewModel::set(IObservableMap<String^, Object^>^ value)
{
    SetValue(DefaultViewModelProperty, value);
}

/// <summary>
/// Invoked when the page is part of the visual tree
/// </summary>
/// <param name="sender">Instance that triggered the event.</param>
/// <param name="e">Event data describing the conditions that led to the event.</param>
void LayoutAwarePage::OnLoaded(Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e)
{
    this->StartLayoutUpdates(sender, e);

    // Keyboard and mouse navigation only apply when occupying the entire window
    if (this->ActualHeight == Window::Current->Bounds.Height &&
        this->ActualWidth == Window::Current->Bounds.Width)
    {
        // Listen to the window directly so focus isn't required
        _acceleratorKeyEventToken = Window::Current->CoreWindow->Dispatcher->AcceleratorKeyActivated +=
            ref new TypedEventHandler<CoreDispatcher^, AcceleratorKeyEventArgs^>(this,
            &LayoutAwarePage::CoreDispatcher_AcceleratorKeyActivated);
        _pointerPressedEventToken = Window::Current->CoreWindow->PointerPressed +=
            ref new TypedEventHandler<CoreWindow^, PointerEventArgs^>(this,
            &LayoutAwarePage::CoreWindow_PointerPressed);
        _navigationShortcutsRegistered = true;
    }
}

/// <summary>
/// Invoked when the page is removed from visual tree
/// </summary>
/// <param name="sender">Instance that triggered the event.</param>
/// <param name="e">Event data describing the conditions that led to the event.</param>
void LayoutAwarePage::OnUnloaded(Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e)
{
    if (_navigationShortcutsRegistered)
    {
        Window::Current->CoreWindow->Dispatcher->AcceleratorKeyActivated -= _acceleratorKeyEventToken;
        Window::Current->CoreWindow->PointerPressed -= _pointerPressedEventToken;
        _navigationShortcutsRegistered = false;
    }
    StopLayoutUpdates(sender, e);
}

#pragma region Navigation support

/// <summary>
/// Invoked as an event handler to navigate backward in the page's associated <see cref="Frame"/>
/// until it reaches the top of the navigation stack.
/// </summary>
/// <param name="sender">Instance that triggered the event.</param>
/// <param name="e">Event data describing the conditions that led to the event.</param>
void LayoutAwarePage::GoHome(Object^ sender, RoutedEventArgs^ e)
{
    (void) sender;	// Unused parameter
    (void) e;	// Unused parameter

    // Use the navigation frame to return to the topmost page
    if (Frame != nullptr)
    {
        while (Frame->CanGoBack)
        {
            Frame->GoBack();
        }
    }
}

/// <summary>
/// Invoked as an event handler to navigate backward in the navigation stack
/// associated with this page's <see cref="Frame"/>.
/// </summary>
/// <param name="sender">Instance that triggered the event.</param>
/// <param name="e">Event data describing the conditions that led to the event.</param>
void LayoutAwarePage::GoBack(Object^ sender, RoutedEventArgs^ e)
{
    (void) sender;	// Unused parameter
    (void) e;	// Unused parameter

    // Use the navigation frame to return to the previous page
    if (Frame != nullptr && Frame->CanGoBack)
    {
        Frame->GoBack();
    }
}

/// <summary>
/// Invoked as an event handler to navigate forward in the navigation stack
/// associated with this page's <see cref="Frame"/>.
/// </summary>
/// <param name="sender">Instance that triggered the event.</param>
/// <param name="e">Event data describing the conditions that led to the event.</param>
void LayoutAwarePage::GoForward(Object^ sender, RoutedEventArgs^ e)
{
    (void) sender;	// Unused parameter
    (void) e;	// Unused parameter

    // Use the navigation frame to advance to the next page
    if (Frame != nullptr && Frame->CanGoForward)
    {
        Frame->GoForward();
    }
}

/// <summary>
/// Invoked on every keystroke, including system keys such as Alt key combinations, when
/// this page is active and occupies the entire window.  Used to detect keyboard navigation
/// between pages even when the page itself doesn't have focus.
/// </summary>
/// <param name="sender">Instance that triggered the event.</param>
/// <param name="args">Event data describing the conditions that led to the event.</param>
void LayoutAwarePage::CoreDispatcher_AcceleratorKeyActivated(CoreDispatcher^ sender, AcceleratorKeyEventArgs^ args)
{
    auto virtualKey = args->VirtualKey;

    // Only investigate further when Left, Right, or the dedicated Previous or Next keys
    // are pressed
    if ((args->EventType == CoreAcceleratorKeyEventType::SystemKeyDown ||
        args->EventType == CoreAcceleratorKeyEventType::KeyDown) &&
        (virtualKey == VirtualKey::Left || virtualKey == VirtualKey::Right ||
        (int)virtualKey == 166 || (int)virtualKey == 167))
    {
        auto coreWindow = Window::Current->CoreWindow;
        auto downState = Windows::UI::Core::CoreVirtualKeyStates::Down;
        bool menuKey = (coreWindow->GetKeyState(VirtualKey::Menu) & downState) == downState;
        bool controlKey = (coreWindow->GetKeyState(VirtualKey::Control) & downState) == downState;
        bool shiftKey = (coreWindow->GetKeyState(VirtualKey::Shift) & downState) == downState;
        bool noModifiers = !menuKey && !controlKey && !shiftKey;
        bool onlyAlt = menuKey && !controlKey && !shiftKey;

        if (((int)virtualKey == 166 && noModifiers) ||
            (virtualKey == VirtualKey::Left && onlyAlt))
        {
            // When the previous key or Alt+Left are pressed navigate back
            args->Handled = true;
            GoBack(this, ref new RoutedEventArgs());
        }
        else if (((int)virtualKey == 167 && noModifiers) ||
            (virtualKey == VirtualKey::Right && onlyAlt))
        {
            // When the next key or Alt+Right are pressed navigate forward
            args->Handled = true;
            GoForward(this, ref new RoutedEventArgs());
        }
    }
}

/// <summary>
/// Invoked on every mouse click, touch screen tap, or equivalent interaction when this
/// page is active and occupies the entire window.  Used to detect browser-style next and
/// previous mouse button clicks to navigate between pages.
/// </summary>
/// <param name="sender">Instance that triggered the event.</param>
/// <param name="args">Event data describing the conditions that led to the event.</param>
void LayoutAwarePage::CoreWindow_PointerPressed(CoreWindow^ sender, PointerEventArgs^ args)
{
    auto properties = args->CurrentPoint->Properties;

    // Ignore button chords with the left, right, and middle buttons
    if (properties->IsLeftButtonPressed || properties->IsRightButtonPressed ||
        properties->IsMiddleButtonPressed) return;

    // If back or forward are pressed (but not both) navigate appropriately
    bool backPressed = properties->IsXButton1Pressed;
    bool forwardPressed = properties->IsXButton2Pressed;
    if (backPressed ^ forwardPressed)
    {
        args->Handled = true;
        if (backPressed) GoBack(this, ref new RoutedEventArgs());
        if (forwardPressed) GoForward(this, ref new RoutedEventArgs());
    }
}

#pragma endregion

#pragma region Visual state switching

/// <summary>
/// Invoked as an event handler, typically on the <see cref="Loaded"/> event of a
/// <see cref="Control"/> within the page, to indicate that the sender should start receiving
/// visual state management changes that correspond to application view state changes.
/// </summary>
/// <param name="sender">Instance of <see cref="Control"/> that supports visual state management
/// corresponding to view states.</param>
/// <param name="e">Event data that describes how the request was made.</param>
/// <remarks>The current view state will immediately be used to set the corresponding visual state
/// when layout updates are requested.  A corresponding <see cref="Unloaded"/> event handler
/// connected to <see cref="StopLayoutUpdates"/> is strongly encouraged.  Instances of
/// <see cref="LayoutAwarePage"/> automatically invoke these handlers in their Loaded and Unloaded
/// events.</remarks>
/// <seealso cref="DetermineVisualState"/>
/// <seealso cref="InvalidateVisualState"/>
void LayoutAwarePage::StartLayoutUpdates(Object^ sender, RoutedEventArgs^ e)
{
    (void) e;	// Unused parameter

    auto control = safe_cast<Control^>(sender);
    if (_layoutAwareControls == nullptr)
    {
        // Start listening to view state changes when there are controls interested in updates
        _layoutAwareControls = ref new Vector<Control^>();
        _windowSizeEventToken = Window::Current->SizeChanged += ref new WindowSizeChangedEventHandler(this, &LayoutAwarePage::WindowSizeChanged);

        // Page receives notifications for children. Protect the page until we stopped layout updates for all controls.
        _this = this;
    }
    _layoutAwareControls->Append(control);

    // Set the initial visual state of the control
    VisualStateManager::GoToState(control, DetermineVisualState(ApplicationView::Value), false);
}

void LayoutAwarePage::WindowSizeChanged(Object^ sender, Windows::UI::Core::WindowSizeChangedEventArgs^ e)
{
    (void) sender;	// Unused parameter
    (void) e;	// Unused parameter

    InvalidateVisualState();
}

/// <summary>
/// Invoked as an event handler, typically on the <see cref="Unloaded"/> event of a
/// <see cref="Control"/>, to indicate that the sender should start receiving visual state
/// management changes that correspond to application view state changes.
/// </summary>
/// <param name="sender">Instance of <see cref="Control"/> that supports visual state management
/// corresponding to view states.</param>
/// <param name="e">Event data that describes how the request was made.</param>
/// <remarks>The current view state will immediately be used to set the corresponding visual state
/// when layout updates are requested.</remarks>
/// <seealso cref="StartLayoutUpdates"/>
void LayoutAwarePage::StopLayoutUpdates(Object^ sender, RoutedEventArgs^ e)
{
    (void) e;	// Unused parameter

    auto control = safe_cast<Control^>(sender);
    unsigned int index;
    if (_layoutAwareControls != nullptr && _layoutAwareControls->IndexOf(control, &index))
    {
        _layoutAwareControls->RemoveAt(index);
        if (_layoutAwareControls->Size == 0)
        {
            // Stop listening to view state changes when no controls are interested in updates
            Window::Current->SizeChanged -= _windowSizeEventToken;
            _layoutAwareControls = nullptr;
            // Last control has received the Unload notification.
            _this = nullptr;
        }
    }
}

/// <summary>
/// Translates <see cref="ApplicationViewState"/> values into strings for visual state management
/// within the page.  The default implementation uses the names of enum values.  Subclasses may
/// override this method to control the mapping scheme used.
/// </summary>
/// <param name="viewState">View state for which a visual state is desired.</param>
/// <returns>Visual state name used to drive the <see cref="VisualStateManager"/></returns>
/// <seealso cref="InvalidateVisualState"/>
String^ LayoutAwarePage::DetermineVisualState(ApplicationViewState viewState)
{
    switch (viewState)
    {
    case ApplicationViewState::Filled:
        return "Filled";
    case ApplicationViewState::Snapped:
        return "Snapped";
    case ApplicationViewState::FullScreenPortrait:
        return "FullScreenPortrait";
    case ApplicationViewState::FullScreenLandscape:
    default:
        return "FullScreenLandscape";
    }
}

/// <summary>
/// Updates all controls that are listening for visual state changes with the correct visual
/// state.
/// </summary>
/// <remarks>
/// Typically used in conjunction with overriding <see cref="DetermineVisualState"/> to
/// signal that a different value may be returned even though the view state has not changed.
/// </remarks>
void LayoutAwarePage::InvalidateVisualState()
{
    if (_layoutAwareControls != nullptr)
    {
        String^ visualState = DetermineVisualState(ApplicationView::Value);
        auto controlIterator = _layoutAwareControls->First();
        while (controlIterator->HasCurrent)
        {
            auto control = controlIterator->Current;
            VisualStateManager::GoToState(control, visualState, false);
            controlIterator->MoveNext();
        }
    }
}

#pragma endregion

#pragma region Process lifetime management

/// <summary>
/// Invoked when this page is about to be displayed in a Frame.
/// </summary>
/// <param name="e">Event data that describes how this page was reached.  The Parameter
/// property provides the group to be displayed.</param>
void LayoutAwarePage::OnNavigatedTo(NavigationEventArgs^ e)
{
    // Returning to a cached page through navigation shouldn't trigger state loading
    if (_pageKey != nullptr) return;

    auto frameState = SuspensionManager::SessionStateForFrame(Frame);
    _pageKey = "Page-" + Frame->BackStackDepth;

    if (e->NavigationMode == NavigationMode::New)
    {
        // Clear existing state for forward navigation when adding a new page to the
        // navigation stack
        auto nextPageKey = _pageKey;
        int nextPageIndex = Frame->BackStackDepth;
        while (frameState->HasKey(nextPageKey))
        {
            frameState->Remove(nextPageKey);
            nextPageIndex++;
            nextPageKey = "Page-" + nextPageIndex;
        }

        // Pass the navigation parameter to the new page
        LoadState(e->Parameter, nullptr);
    }
    else
    {
        // Pass the navigation parameter and preserved page state to the page, using
        // the same strategy for loading suspended state and recreating pages discarded
        // from cache
        LoadState(e->Parameter, safe_cast<IMap<String^, Object^>^>(frameState->Lookup(_pageKey)));
    }
}

/// <summary>
/// Invoked when this page will no longer be displayed in a Frame.
/// </summary>
/// <param name="e">Event data that describes how this page was reached.  The Parameter
/// property provides the group to be displayed.</param>
void LayoutAwarePage::OnNavigatedFrom(NavigationEventArgs^ e)
{
    auto frameState = SuspensionManager::SessionStateForFrame(Frame);
    auto pageState = ref new Map<String^, Object^>();
    SaveState(pageState);
    frameState->Insert(_pageKey, pageState);
}

/// <summary>
/// Populates the page with content passed during navigation.  Any saved state is also
/// provided when recreating a page from a prior session.
/// </summary>
/// <param name="navigationParameter">The parameter value passed to
/// <see cref="Frame.Navigate(Type, Object)"/> when this page was initially requested.
/// </param>
/// <param name="pageState">A map of state preserved by this page during an earlier
/// session.  This will be null the first time a page is visited.</param>
void LayoutAwarePage::LoadState(Object^ navigationParameter, IMap<String^, Object^>^ pageState)
{
}

/// <summary>
/// Preserves state associated with this page in case the application is suspended or the
/// page is discarded from the navigation cache.  Values must conform to the serialization
/// requirements of <see cref="SuspensionManager.SessionState"/>.
/// </summary>
/// <param name="pageState">An empty map to be populated with serializable state.</param>
void LayoutAwarePage::SaveState(IMap<String^, Object^>^ pageState)
{
}

#pragma endregion
