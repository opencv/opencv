#pragma once

#include <string>
#include <opencv2/viz/types.hpp>

namespace cv
{
    namespace viz
    {
        class KeyboardEvent
        {
        public:
            static const unsigned int Alt   = 1;
            static const unsigned int Ctrl  = 2;
            static const unsigned int Shift = 4;

            /** \brief Constructor
              * \param[in] action    true for key was pressed, false for released
              * \param[in] key_sym   the key-name that caused the action
              * \param[in] key       the key code that caused the action
              * \param[in] alt       whether the alt key was pressed at the time where this event was triggered
              * \param[in] ctrl      whether the ctrl was pressed at the time where this event was triggered
              * \param[in] shift     whether the shift was pressed at the time where this event was triggered
              */
            KeyboardEvent (bool action, const std::string& key_sym, unsigned char key, bool alt, bool ctrl, bool shift);

            bool isAltPressed () const;
            bool isCtrlPressed () const;
            bool isShiftPressed () const;

            unsigned char getKeyCode () const;

            const String& getKeySym () const;
            bool keyDown () const;
            bool keyUp () const;

        protected:

            bool action_;
            unsigned int modifiers_;
            unsigned char key_code_;
            String key_sym_;
        };

        class MouseEvent
        {
        public:
            enum Type
            {
                MouseMove = 1,
                MouseButtonPress,
                MouseButtonRelease,
                MouseScrollDown,
                MouseScrollUp,
                MouseDblClick
            } ;

            enum MouseButton
            {
                NoButton      = 0,
                LeftButton,
                MiddleButton,
                RightButton,
                VScroll /*other buttons, scroll wheels etc. may follow*/
            } ;

            MouseEvent (const Type& type, const MouseButton& button, const Point& p, bool alt, bool ctrl, bool shift);


            Type type;
            MouseButton button;
            Point pointer;
            unsigned int key_state;
        };
    }
}

////////////////////////////////////////////////////////////////////
/// Implementation

inline cv::viz::KeyboardEvent::KeyboardEvent (bool _action, const std::string& _key_sym, unsigned char key, bool alt, bool ctrl, bool shift)
  : action_ (_action), modifiers_ (0), key_code_(key), key_sym_ (_key_sym)
{
  if (alt)
    modifiers_ = Alt;

  if (ctrl)
    modifiers_ |= Ctrl;

  if (shift)
    modifiers_ |= Shift;
}

inline bool cv::viz::KeyboardEvent::isAltPressed () const { return (modifiers_ & Alt) != 0; }
inline bool cv::viz::KeyboardEvent::isCtrlPressed () const { return (modifiers_ & Ctrl) != 0; }
inline bool cv::viz::KeyboardEvent::isShiftPressed () const { return (modifiers_ & Shift) != 0; }
inline unsigned char cv::viz::KeyboardEvent::getKeyCode () const { return key_code_; }
inline const cv::String& cv::viz::KeyboardEvent::getKeySym () const { return key_sym_; }
inline bool cv::viz::KeyboardEvent::keyDown () const { return action_; }
inline bool cv::viz::KeyboardEvent::keyUp () const { return !action_; }

inline cv::viz::MouseEvent::MouseEvent (const Type& _type, const MouseButton& _button, const Point& _p,  bool alt, bool ctrl, bool shift)
    : type(_type), button(_button), pointer(_p), key_state(0)
{
    if (alt)
        key_state = KeyboardEvent::Alt;

    if (ctrl)
        key_state |= KeyboardEvent::Ctrl;

    if (shift)
        key_state |= KeyboardEvent::Shift;
}
