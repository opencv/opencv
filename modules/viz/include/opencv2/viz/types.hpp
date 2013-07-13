#pragma once

#include <string>
#include <opencv2/core.hpp>
#include <opencv2/core/affine.hpp>

namespace cv
{
    typedef std::string String;

    namespace viz
    {
        class CV_EXPORTS Color : public Scalar
        {
        public:
            Color();
            Color(double gray);
            Color(double blue, double green, double red);

            Color(const Scalar& color);

            static Color black();
            static Color blue();
            static Color green();
            static Color cyan();

            static Color red();
            static Color magenta();
            static Color yellow();
            static Color white();

            static Color gray();
        };


        struct CV_EXPORTS Vertices
        {
            std::vector<unsigned int> vertices;
        };

        class CV_EXPORTS Mesh3d
        {
        public:
            typedef Ptr<Mesh3d> Ptr;

            Mat cloud, colors;
            std::vector<Vertices> polygons;

            static Mesh3d::Ptr mesh_load(const String& file);

        };

        class CV_EXPORTS KeyboardEvent
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

        class CV_EXPORTS MouseEvent
        {
        public:
            enum Type { MouseMove = 1, MouseButtonPress, MouseButtonRelease, MouseScrollDown, MouseScrollUp, MouseDblClick } ;
            enum MouseButton { NoButton = 0, LeftButton, MiddleButton, RightButton, VScroll } ;

            MouseEvent (const Type& type, const MouseButton& button, const Point& p, bool alt, bool ctrl, bool shift);

            Type type;
            MouseButton button;
            Point pointer;
            unsigned int key_state;
        };

    } /* namespace viz */
} /* namespace cv */




