#include <opencv2/viz/types.hpp>

//////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::viz::Color

cv::viz::Color::Color() : Scalar(0, 0, 0) {}
cv::viz::Color::Color(double gray) : Scalar(gray, gray, gray) {}
cv::viz::Color::Color(double blue, double green, double red) : Scalar(blue, green, red) {}
cv::viz::Color::Color(const Scalar& color) : Scalar(color) {}

cv::viz::Color cv::viz::Color::black()   { return Color(  0,   0, 0); }
cv::viz::Color cv::viz::Color::green()   { return Color(  0, 255, 0); }
cv::viz::Color cv::viz::Color::blue()    { return Color(255,   0, 0); }
cv::viz::Color cv::viz::Color::cyan()    { return Color(255, 255, 0); }

cv::viz::Color cv::viz::Color::red()     { return Color(  0,   0, 255); }
cv::viz::Color cv::viz::Color::magenta() { return Color(  0, 255, 255); }
cv::viz::Color cv::viz::Color::yellow()  { return Color(255,   0, 255); }
cv::viz::Color cv::viz::Color::white()   { return Color(255, 255, 255); }

cv::viz::Color cv::viz::Color::gray()    { return Color(128, 128, 128); }

////////////////////////////////////////////////////////////////////
/// cv::viz::KeyboardEvent

cv::viz::KeyboardEvent::KeyboardEvent (bool _action, const std::string& _key_sym, unsigned char key, bool alt, bool ctrl, bool shift)
  : action_ (_action), modifiers_ (0), key_code_(key), key_sym_ (_key_sym)
{
  if (alt)
    modifiers_ = Alt;

  if (ctrl)
    modifiers_ |= Ctrl;

  if (shift)
    modifiers_ |= Shift;
}

bool cv::viz::KeyboardEvent::isAltPressed () const { return (modifiers_ & Alt) != 0; }
bool cv::viz::KeyboardEvent::isCtrlPressed () const { return (modifiers_ & Ctrl) != 0; }
bool cv::viz::KeyboardEvent::isShiftPressed () const { return (modifiers_ & Shift) != 0; }
unsigned char cv::viz::KeyboardEvent::getKeyCode () const { return key_code_; }
const cv::String& cv::viz::KeyboardEvent::getKeySym () const { return key_sym_; }
bool cv::viz::KeyboardEvent::keyDown () const { return action_; }
bool cv::viz::KeyboardEvent::keyUp () const { return !action_; }

////////////////////////////////////////////////////////////////////
/// cv::viz::MouseEvent

cv::viz::MouseEvent::MouseEvent (const Type& _type, const MouseButton& _button, const Point& _p,  bool alt, bool ctrl, bool shift)
    : type(_type), button(_button), pointer(_p), key_state(0)
{
    if (alt)
        key_state = KeyboardEvent::Alt;

    if (ctrl)
        key_state |= KeyboardEvent::Ctrl;

    if (shift)
        key_state |= KeyboardEvent::Shift;
}
