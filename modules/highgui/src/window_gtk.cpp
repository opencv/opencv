/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

#ifndef WIN32

#ifdef HAVE_GTK

#include "gtk/gtk.h"
#include "gdk/gdkkeysyms.h"
#include <stdio.h>

#ifdef HAVE_OPENGL
    #include <gtk/gtkgl.h>
    #include <GL/gl.h>
    #include <GL/glu.h>
#endif

// TODO Fix the initial window size when flags=0.  Right now the initial window is by default
// 320x240 size.  A better default would be actual size of the image.  Problem
// is determining desired window size with trackbars while still allowing resizing.
//
// Gnome Totem source may be of use here, see bacon_video_widget_set_scale_ratio
// in totem/src/backend/bacon-video-widget-xine.c

////////////////////////////////////////////////////////////
// CvImageWidget GTK Widget Public API
////////////////////////////////////////////////////////////
typedef struct _CvImageWidget        CvImageWidget;
typedef struct _CvImageWidgetClass   CvImageWidgetClass;

struct _CvImageWidget {
    GtkWidget widget;
    CvMat * original_image;
    CvMat * scaled_image;
    int flags;
};

struct _CvImageWidgetClass
{
  GtkWidgetClass parent_class;
};


/** Allocate new image viewer widget */
GtkWidget*     cvImageWidgetNew      (int flags);

/** Set the image to display in the widget */
void           cvImageWidgetSetImage(CvImageWidget * widget, const CvArr *arr);

// standard GTK object macros
#define CV_IMAGE_WIDGET(obj)          GTK_CHECK_CAST (obj, cvImageWidget_get_type (), CvImageWidget)
#define CV_IMAGE_WIDGET_CLASS(klass)  GTK_CHECK_CLASS_CAST (klass, cvImageWidget_get_type (), CvImageWidgetClass)
#define CV_IS_IMAGE_WIDGET(obj)       GTK_CHECK_TYPE (obj, cvImageWidget_get_type ())

/////////////////////////////////////////////////////////////////////////////
// Private API ////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
GtkType        cvImageWidget_get_type (void);

static GtkWidgetClass * parent_class = NULL;

// flag to help size initial window
#define CV_WINDOW_NO_IMAGE 2

void cvImageWidgetSetImage(CvImageWidget * widget, const CvArr *arr){
    CvMat * mat, stub;
    int origin=0;

    //printf("cvImageWidgetSetImage\n");

    if( CV_IS_IMAGE_HDR( arr ))
        origin = ((IplImage*)arr)->origin;

    mat = cvGetMat(arr, &stub);

    if(widget->original_image && !CV_ARE_SIZES_EQ(mat, widget->original_image)){
        cvReleaseMat( &widget->original_image );
    }
    if(!widget->original_image){
        widget->original_image = cvCreateMat( mat->rows, mat->cols, CV_8UC3 );
        gtk_widget_queue_resize( GTK_WIDGET( widget ) );
    }
    cvConvertImage( mat, widget->original_image,
                            (origin != 0 ? CV_CVTIMG_FLIP : 0) + CV_CVTIMG_SWAP_RB );
    if(widget->scaled_image){
        cvResize( widget->original_image, widget->scaled_image, CV_INTER_AREA );
    }

    // window does not refresh without this
    gtk_widget_queue_draw( GTK_WIDGET(widget) );
}

GtkWidget*
cvImageWidgetNew (int flags)
{
  CvImageWidget *image_widget;

  image_widget = CV_IMAGE_WIDGET( gtk_type_new (cvImageWidget_get_type ()) );
  image_widget->original_image = 0;
  image_widget->scaled_image = 0;
  image_widget->flags = flags | CV_WINDOW_NO_IMAGE;

  return GTK_WIDGET (image_widget);
}

static void
cvImageWidget_realize (GtkWidget *widget)
{
  GdkWindowAttr attributes;
  gint attributes_mask;

  //printf("cvImageWidget_realize\n");
  g_return_if_fail (widget != NULL);
  g_return_if_fail (CV_IS_IMAGE_WIDGET (widget));

  GTK_WIDGET_SET_FLAGS (widget, GTK_REALIZED);

  attributes.x = widget->allocation.x;
  attributes.y = widget->allocation.y;
  attributes.width = widget->allocation.width;
  attributes.height = widget->allocation.height;
  attributes.wclass = GDK_INPUT_OUTPUT;
  attributes.window_type = GDK_WINDOW_CHILD;
  attributes.event_mask = gtk_widget_get_events (widget) |
    GDK_EXPOSURE_MASK | GDK_BUTTON_PRESS_MASK |
    GDK_BUTTON_RELEASE_MASK | GDK_POINTER_MOTION_MASK;
  attributes.visual = gtk_widget_get_visual (widget);
  attributes.colormap = gtk_widget_get_colormap (widget);

  attributes_mask = GDK_WA_X | GDK_WA_Y | GDK_WA_VISUAL | GDK_WA_COLORMAP;
  widget->window = gdk_window_new (widget->parent->window, &attributes, attributes_mask);

  widget->style = gtk_style_attach (widget->style, widget->window);

  gdk_window_set_user_data (widget->window, widget);

  gtk_style_set_background (widget->style, widget->window, GTK_STATE_ACTIVE);
}

static CvSize cvImageWidget_calc_size( int im_width, int im_height, int max_width, int max_height ){
    float aspect = (float)im_width/(float)im_height;
    float max_aspect = (float)max_width/(float)max_height;
    if(aspect > max_aspect){
        return cvSize( max_width, cvRound(max_width/aspect) );
    }
    return cvSize( cvRound(max_height*aspect), max_height );
}

static void
cvImageWidget_size_request (GtkWidget      *widget,
                       GtkRequisition *requisition)
{
    CvImageWidget * image_widget = CV_IMAGE_WIDGET( widget );

    //printf("cvImageWidget_size_request ");
    // the case the first time cvShowImage called or when AUTOSIZE
    if( image_widget->original_image &&
        ((image_widget->flags & CV_WINDOW_AUTOSIZE) ||
         (image_widget->flags & CV_WINDOW_NO_IMAGE)))
    {
        //printf("original ");
        requisition->width = image_widget->original_image->cols;
        requisition->height = image_widget->original_image->rows;
    }
    // default case
    else if(image_widget->scaled_image){
        //printf("scaled ");
        requisition->width = image_widget->scaled_image->cols;
        requisition->height = image_widget->scaled_image->rows;
    }
    // the case before cvShowImage called
    else{
        //printf("default ");
        requisition->width = 320;
        requisition->height = 240;
    }
    //printf("%d %d\n",requisition->width, requisition->height);
}

static void cvImageWidget_set_size(GtkWidget * widget, int max_width, int max_height){
    CvImageWidget * image_widget = CV_IMAGE_WIDGET( widget );

    //printf("cvImageWidget_set_size %d %d\n", max_width, max_height);

    // don't allow to set the size
    if(image_widget->flags & CV_WINDOW_AUTOSIZE) return;
    if(!image_widget->original_image) return;

    CvSize scaled_image_size = cvImageWidget_calc_size( image_widget->original_image->cols,
            image_widget->original_image->rows, max_width, max_height );

    if( image_widget->scaled_image &&
            ( image_widget->scaled_image->cols != scaled_image_size.width ||
              image_widget->scaled_image->rows != scaled_image_size.height ))
    {
        cvReleaseMat( &image_widget->scaled_image );
    }
    if( !image_widget->scaled_image ){
        image_widget->scaled_image = cvCreateMat( scaled_image_size.height, scaled_image_size.width,        CV_8UC3 );


    }
    assert( image_widget->scaled_image );
}

static void
cvImageWidget_size_allocate (GtkWidget     *widget,
                        GtkAllocation *allocation)
{
  CvImageWidget *image_widget;

  //printf("cvImageWidget_size_allocate\n");
  g_return_if_fail (widget != NULL);
  g_return_if_fail (CV_IS_IMAGE_WIDGET (widget));
  g_return_if_fail (allocation != NULL);

  widget->allocation = *allocation;
  image_widget = CV_IMAGE_WIDGET (widget);


  if( (image_widget->flags & CV_WINDOW_AUTOSIZE)==0 && image_widget->original_image ){
      // (re) allocated scaled image
      if( image_widget->flags & CV_WINDOW_NO_IMAGE ){
          cvImageWidget_set_size( widget, image_widget->original_image->cols,
                                          image_widget->original_image->rows);
      }
      else{
          cvImageWidget_set_size( widget, allocation->width, allocation->height );
      }
      cvResize( image_widget->original_image, image_widget->scaled_image, CV_INTER_AREA );
  }

  if (GTK_WIDGET_REALIZED (widget))
    {
      image_widget = CV_IMAGE_WIDGET (widget);

      if( image_widget->original_image &&
              ((image_widget->flags & CV_WINDOW_AUTOSIZE) ||
               (image_widget->flags & CV_WINDOW_NO_IMAGE)) )
      {
          widget->allocation.width = image_widget->original_image->cols;
          widget->allocation.height = image_widget->original_image->rows;
          gdk_window_move_resize( widget->window, allocation->x, allocation->y,
                  image_widget->original_image->cols, image_widget->original_image->rows );
          if(image_widget->flags & CV_WINDOW_NO_IMAGE){
              image_widget->flags &= ~CV_WINDOW_NO_IMAGE;
              gtk_widget_queue_resize( GTK_WIDGET(widget) );
          }
      }
      else{
          gdk_window_move_resize (widget->window,
                  allocation->x, allocation->y,
                  allocation->width, allocation->height );

      }
    }
}

static void
cvImageWidget_destroy (GtkObject *object)
{
  CvImageWidget *image_widget;

  g_return_if_fail (object != NULL);
  g_return_if_fail (CV_IS_IMAGE_WIDGET (object));

  image_widget = CV_IMAGE_WIDGET (object);

  cvReleaseMat( &image_widget->scaled_image );
  cvReleaseMat( &image_widget->original_image );

  if (GTK_OBJECT_CLASS (parent_class)->destroy)
    (* GTK_OBJECT_CLASS (parent_class)->destroy) (object);
}

static void cvImageWidget_class_init (CvImageWidgetClass * klass)
{
  GtkObjectClass *object_class;
  GtkWidgetClass *widget_class;

  object_class = (GtkObjectClass*) klass;
  widget_class = (GtkWidgetClass*) klass;

  parent_class = GTK_WIDGET_CLASS( gtk_type_class (gtk_widget_get_type ()) );

  object_class->destroy = cvImageWidget_destroy;

  widget_class->realize = cvImageWidget_realize;
  widget_class->size_request = cvImageWidget_size_request;
  widget_class->size_allocate = cvImageWidget_size_allocate;
  widget_class->button_press_event = NULL;
  widget_class->button_release_event = NULL;
  widget_class->motion_notify_event = NULL;
}

static void
cvImageWidget_init (CvImageWidget *image_widget)
{
    image_widget->original_image=0;
    image_widget->scaled_image=0;
    image_widget->flags=0;
}

GtkType cvImageWidget_get_type (void){
  static GtkType image_type = 0;

  if (!image_type)
    {
      static const GtkTypeInfo image_info =
      {
        (gchar*)"CvImageWidget",
        sizeof (CvImageWidget),
        sizeof (CvImageWidgetClass),
        (GtkClassInitFunc) cvImageWidget_class_init,
        (GtkObjectInitFunc) cvImageWidget_init,
        /* reserved_1 */ NULL,
        /* reserved_1 */ NULL,
        (GtkClassInitFunc) NULL
      };

      image_type = gtk_type_unique (GTK_TYPE_WIDGET, &image_info);
    }

  return image_type;
}
/////////////////////////////////////////////////////////////////////////////
// End CvImageWidget
/////////////////////////////////////////////////////////////////////////////


struct CvWindow;

typedef struct CvTrackbar
{
    int signature;
    GtkWidget* widget;
    char* name;
    CvTrackbar* next;
    CvWindow* parent;
    int* data;
    int pos;
    int maxval;
    CvTrackbarCallback notify;
    CvTrackbarCallback2 notify2;
    void* userdata;
}
CvTrackbar;


typedef struct CvWindow
{
    int signature;
    GtkWidget* widget;
    GtkWidget* frame;
    GtkWidget* paned;
    char* name;
    CvWindow* prev;
    CvWindow* next;

    int last_key;
    int flags;
    int status;//0 normal, 1 fullscreen (YV)

    CvMouseCallback on_mouse;
    void* on_mouse_param;

    struct
    {
        int pos;
        int rows;
        CvTrackbar* first;
    }
    toolbar;

#ifdef HAVE_OPENGL
    bool useGl;

    CvOpenGlDrawCallback glDrawCallback;
    void* glDrawData;

    CvOpenGlCleanCallback glCleanCallback;
    void* glCleanData;
#endif
}
CvWindow;


static gboolean icvOnClose( GtkWidget* widget, GdkEvent* event, gpointer user_data );
static gboolean icvOnKeyPress( GtkWidget* widget, GdkEventKey* event, gpointer user_data );
static void icvOnTrackbar( GtkWidget* widget, gpointer user_data );
static gboolean icvOnMouse( GtkWidget *widget, GdkEvent *event, gpointer user_data );

#ifdef HAVE_GTHREAD
int thread_started=0;
static gpointer icvWindowThreadLoop();
GMutex*				   last_key_mutex;
GCond*				   cond_have_key;
GMutex*				   window_mutex;
GThread*			   window_thread;
GtkWidget*             cvTopLevelWidget = 0;
#endif

static int             last_key = -1;
static CvWindow* hg_windows = 0;

CV_IMPL int cvInitSystem( int argc, char** argv )
{
    static int wasInitialized = 0;

    // check initialization status
    if( !wasInitialized )
    {
        hg_windows = 0;

        gtk_disable_setlocale();
        gtk_init( &argc, &argv );

        #ifdef HAVE_OPENGL
            gtk_gl_init(&argc, &argv);
        #endif

        wasInitialized = 1;
    }

    return 0;
}

CV_IMPL int cvStartWindowThread(){
#ifdef HAVE_GTHREAD
    cvInitSystem(0,NULL);
    if (!thread_started) {
    if (!g_thread_supported ()) {
        /* the GThread system wasn't inited, so init it */
        g_thread_init(NULL);
    }

    // this mutex protects the window resources
    window_mutex = g_mutex_new();

    // protects the 'last key pressed' variable
    last_key_mutex = g_mutex_new();

    // conditional that indicates a key has been pressed
    cond_have_key = g_cond_new();

    // this is the window update thread
    window_thread = g_thread_create((GThreadFunc) icvWindowThreadLoop,
                    NULL, TRUE, NULL);
    }
    thread_started = window_thread!=NULL;
    return thread_started;
#else
    return 0;
#endif
}

#ifdef HAVE_GTHREAD
gpointer icvWindowThreadLoop(){
    while(1){
        g_mutex_lock(window_mutex);
        gtk_main_iteration_do(FALSE);
        g_mutex_unlock(window_mutex);

        // little sleep
        g_usleep(500);

        g_thread_yield();
    }
    return NULL;
}

#define CV_LOCK_MUTEX() \
if(thread_started && g_thread_self()!=window_thread){ g_mutex_lock( window_mutex ); } else { }

#define CV_UNLOCK_MUTEX() \
if(thread_started && g_thread_self()!=window_thread){ g_mutex_unlock( window_mutex); } else { }

#else
#define CV_LOCK_MUTEX()
#define CV_UNLOCK_MUTEX()
#endif

static CvWindow* icvFindWindowByName( const char* name )
{
    CvWindow* window = hg_windows;
    while( window != 0 && strcmp(name, window->name) != 0 )
        window = window->next;

    return window;
}

static CvWindow* icvWindowByWidget( GtkWidget* widget )
{
    CvWindow* window = hg_windows;

    while( window != 0 && window->widget != widget &&
           window->frame != widget && window->paned != widget )
        window = window->next;

    return window;
}

double cvGetModeWindow_GTK(const char* name)//YV
{
    double result = -1;

    CV_FUNCNAME( "cvGetModeWindow_GTK" );

    __BEGIN__;

    CvWindow* window;

    if (!name)
        CV_ERROR( CV_StsNullPtr, "NULL name string" );

    window = icvFindWindowByName( name );
    if (!window)
        CV_ERROR( CV_StsNullPtr, "NULL window" );

    CV_LOCK_MUTEX();
    result = window->status;
    CV_UNLOCK_MUTEX();

    __END__;
    return result;
}


void cvSetModeWindow_GTK( const char* name, double prop_value)//Yannick Verdie
{

    CV_FUNCNAME( "cvSetModeWindow_GTK" );

    __BEGIN__;

    CvWindow* window;

    if(!name)
        CV_ERROR( CV_StsNullPtr, "NULL name string" );

    window = icvFindWindowByName( name );
    if( !window )
        CV_ERROR( CV_StsNullPtr, "NULL window" );

    if(window->flags & CV_WINDOW_AUTOSIZE)//if the flag CV_WINDOW_AUTOSIZE is set
        EXIT;

    //so easy to do fullscreen here, Linux rocks !

    if (window->status==CV_WINDOW_FULLSCREEN && prop_value==CV_WINDOW_NORMAL)
    {
        CV_LOCK_MUTEX();
        gtk_window_unfullscreen(GTK_WINDOW(window->frame));
        window->status=CV_WINDOW_NORMAL;
        CV_UNLOCK_MUTEX();
        EXIT;
    }

    if (window->status==CV_WINDOW_NORMAL && prop_value==CV_WINDOW_FULLSCREEN)
    {
        CV_LOCK_MUTEX();
        gtk_window_fullscreen(GTK_WINDOW(window->frame));
        window->status=CV_WINDOW_FULLSCREEN;
        CV_UNLOCK_MUTEX();
        EXIT;
    }

    __END__;
}


double cvGetPropWindowAutoSize_GTK(const char* name)
{
    double result = -1;

    CV_FUNCNAME( "cvGetPropWindowAutoSize_GTK" );

    __BEGIN__;

    CvWindow* window;

    if (!name)
        CV_ERROR( CV_StsNullPtr, "NULL name string" );

    window = icvFindWindowByName( name );
    if (!window)
        EXIT; // keep silence here

    result = window->flags & CV_WINDOW_AUTOSIZE;

    __END__;

    return result;
}

double cvGetRatioWindow_GTK(const char* name)
{
    double result = -1;

    CV_FUNCNAME( "cvGetRatioWindow_GTK" );

    __BEGIN__;

    CvWindow* window;

    if (!name)
        CV_ERROR( CV_StsNullPtr, "NULL name string" );

    window = icvFindWindowByName( name );
    if (!window)
        EXIT; // keep silence here

    result = static_cast<double>(window->widget->allocation.width) / window->widget->allocation.height;

    __END__;

    return result;
}

double cvGetOpenGlProp_GTK(const char* name)
{
    double result = -1;

#ifdef HAVE_OPENGL
    CV_FUNCNAME( "cvGetOpenGlProp_GTK" );

    __BEGIN__;

    CvWindow* window;

    if (!name)
        CV_ERROR( CV_StsNullPtr, "NULL name string" );

    window = icvFindWindowByName( name );
    if (!window)
        EXIT; // keep silence here

    result = window->useGl;

    __END__;
#endif

    return result;
}


// OpenGL support

#ifdef HAVE_OPENGL

namespace
{
    class GlFuncTab_GTK : public CvOpenGlFuncTab
    {
    public:
        GlFuncTab_GTK();

        void genBuffers(int n, unsigned int* buffers) const;
        void deleteBuffers(int n, const unsigned int* buffers) const;

        void bufferData(unsigned int target, ptrdiff_t size, const void* data, unsigned int usage) const;
        void bufferSubData(unsigned int target, ptrdiff_t offset, ptrdiff_t size, const void* data) const;

        void bindBuffer(unsigned int target, unsigned int buffer) const;

        void* mapBuffer(unsigned int target, unsigned int access) const;
        void unmapBuffer(unsigned int target) const;

        void generateBitmapFont(const std::string& family, int height, int weight, bool italic, bool underline, int start, int count, int base) const;

        bool isGlContextInitialized() const;

        PFNGLGENBUFFERSPROC    glGenBuffersExt;
        PFNGLDELETEBUFFERSPROC glDeleteBuffersExt;

        PFNGLBUFFERDATAPROC    glBufferDataExt;
        PFNGLBUFFERSUBDATAPROC glBufferSubDataExt;

        PFNGLBINDBUFFERPROC    glBindBufferExt;

        PFNGLMAPBUFFERPROC     glMapBufferExt;
        PFNGLUNMAPBUFFERPROC   glUnmapBufferExt;

        bool initialized;
    };

    GlFuncTab_GTK::GlFuncTab_GTK()
    {
        glGenBuffersExt    = 0;
        glDeleteBuffersExt = 0;

        glBufferDataExt    = 0;
        glBufferSubDataExt = 0;

        glBindBufferExt    = 0;

        glMapBufferExt     = 0;
        glUnmapBufferExt   = 0;

        initialized = false;
    }

    void GlFuncTab_GTK::genBuffers(int n, unsigned int* buffers) const
    {
        CV_FUNCNAME( "GlFuncTab_GTK::genBuffers" );

        __BEGIN__;

        if (!glGenBuffersExt)
            CV_ERROR(CV_OpenGlApiCallError, "Current OpenGL implementation doesn't support required extension");

        glGenBuffersExt(n, buffers);
        CV_CheckGlError();

        __END__;
    }

    void GlFuncTab_GTK::deleteBuffers(int n, const unsigned int* buffers) const
    {
        CV_FUNCNAME( "GlFuncTab_GTK::deleteBuffers" );

        __BEGIN__;

        if (!glDeleteBuffersExt)
            CV_ERROR(CV_OpenGlApiCallError, "Current OpenGL implementation doesn't support required extension");

        glDeleteBuffersExt(n, buffers);
        CV_CheckGlError();

        __END__;
    }

    void GlFuncTab_GTK::bufferData(unsigned int target, ptrdiff_t size, const void* data, unsigned int usage) const
    {
        CV_FUNCNAME( "GlFuncTab_GTK::bufferData" );

        __BEGIN__;

        if (!glBufferDataExt)
            CV_ERROR(CV_OpenGlApiCallError, "Current OpenGL implementation doesn't support required extension");

        glBufferDataExt(target, size, data, usage);
        CV_CheckGlError();

        __END__;
    }

    void GlFuncTab_GTK::bufferSubData(unsigned int target, ptrdiff_t offset, ptrdiff_t size, const void* data) const
    {
        CV_FUNCNAME( "GlFuncTab_GTK::bufferSubData" );

        __BEGIN__;

        if (!glBufferSubDataExt)
            CV_ERROR(CV_OpenGlApiCallError, "Current OpenGL implementation doesn't support required extension");

        glBufferSubDataExt(target, offset, size, data);
        CV_CheckGlError();

        __END__;
    }

    void GlFuncTab_GTK::bindBuffer(unsigned int target, unsigned int buffer) const
    {
        CV_FUNCNAME( "GlFuncTab_GTK::bindBuffer" );

        __BEGIN__;

        if (!glBindBufferExt)
            CV_ERROR(CV_OpenGlApiCallError, "Current OpenGL implementation doesn't support required extension");

        glBindBufferExt(target, buffer);
        CV_CheckGlError();

        __END__;
    }

    void* GlFuncTab_GTK::mapBuffer(unsigned int target, unsigned int access) const
    {
        CV_FUNCNAME( "GlFuncTab_GTK::mapBuffer" );

        void* res = 0;

        __BEGIN__;

        if (!glMapBufferExt)
            CV_ERROR(CV_OpenGlApiCallError, "Current OpenGL implementation doesn't support required extension");

        res = glMapBufferExt(target, access);
        CV_CheckGlError();

        __END__;

        return res;
    }

    void GlFuncTab_GTK::unmapBuffer(unsigned int target) const
    {
        CV_FUNCNAME( "GlFuncTab_GTK::unmapBuffer" );

        __BEGIN__;

        if (!glUnmapBufferExt)
            CV_ERROR(CV_OpenGlApiCallError, "Current OpenGL implementation doesn't support required extension");

        glUnmapBufferExt(target);
        CV_CheckGlError();

        __END__;
    }

    void GlFuncTab_GTK::generateBitmapFont(const std::string& family, int height, int weight, bool italic, bool underline, int start, int count, int base) const
    {
        PangoFontDescription* fontDecr;
        PangoFont* pangoFont;

        CV_FUNCNAME( "GlFuncTab_GTK::generateBitmapFont" );

        __BEGIN__;

        fontDecr = pango_font_description_new();

        pango_font_description_set_size(fontDecr, height);

        pango_font_description_set_family_static(fontDecr, family.c_str());

        pango_font_description_set_weight(fontDecr, static_cast<PangoWeight>(weight));

        pango_font_description_set_style(fontDecr, italic ? PANGO_STYLE_ITALIC : PANGO_STYLE_NORMAL);

        pangoFont = gdk_gl_font_use_pango_font(fontDecr, start, count, base);

        pango_font_description_free(fontDecr);

        if (!pangoFont)
            CV_ERROR(CV_OpenGlApiCallError, "Can't create font");

        __END__;
    }

    bool GlFuncTab_GTK::isGlContextInitialized() const
    {
        return initialized;
    }

    void initGl()
    {
        static GlFuncTab_GTK glFuncTab;
        static bool first = true;

        if (first)
        {
            // Load extensions
            GdkGLProc func;

            func = gdk_gl_get_proc_address("glGenBuffers");
            glFuncTab.glGenBuffersExt = (PFNGLGENBUFFERSPROC)func;

            func = gdk_gl_get_proc_address("glDeleteBuffers");
            glFuncTab.glDeleteBuffersExt = (PFNGLDELETEBUFFERSPROC)func;

            func = gdk_gl_get_proc_address("glBufferData");
            glFuncTab.glBufferDataExt = (PFNGLBUFFERDATAPROC)func;

            func = gdk_gl_get_proc_address("glBufferSubData");
            glFuncTab.glBufferSubDataExt = (PFNGLBUFFERSUBDATAPROC)func;

            func = gdk_gl_get_proc_address("glBindBuffer");
            glFuncTab.glBindBufferExt = (PFNGLBINDBUFFERPROC)func;

            func = gdk_gl_get_proc_address("glMapBuffer");
            glFuncTab.glMapBufferExt = (PFNGLMAPBUFFERPROC)func;

            func = gdk_gl_get_proc_address("glUnmapBuffer");
            glFuncTab.glUnmapBufferExt = (PFNGLUNMAPBUFFERPROC)func;

            glFuncTab.initialized = true;

            icvSetOpenGlFuncTab(&glFuncTab);

            first = false;
        }
    }

    void createGlContext(CvWindow* window)
    {
        GdkGLConfig* glconfig;

        CV_FUNCNAME( "createGlContext" );

        __BEGIN__;

        window->useGl = false;

        // Try double-buffered visual
        glconfig = gdk_gl_config_new_by_mode((GdkGLConfigMode)(GDK_GL_MODE_RGB | GDK_GL_MODE_DEPTH | GDK_GL_MODE_DOUBLE));
        if (!glconfig)
            CV_ERROR( CV_OpenGlApiCallError, "Can't Create A GL Device Context" );

        // Set OpenGL-capability to the widget
        if (!gtk_widget_set_gl_capability(window->widget, glconfig, NULL, TRUE, GDK_GL_RGBA_TYPE))
            CV_ERROR( CV_OpenGlApiCallError, "Can't Create A GL Device Context" );

        initGl();

        window->useGl = true;

        __END__;
    }

    void releaseGlContext(CvWindow* window)
    {
        //CV_FUNCNAME( "releaseGlContext" );

        //__BEGIN__;

        window->useGl = false;

        //__END__;
    }

    void drawGl(CvWindow* window)
    {
        CV_FUNCNAME( "drawGl" );

        __BEGIN__;

        GdkGLContext* glcontext = gtk_widget_get_gl_context(window->widget);
        GdkGLDrawable* gldrawable = gtk_widget_get_gl_drawable(window->widget);

        if (!gdk_gl_drawable_gl_begin (gldrawable, glcontext))
            CV_ERROR( CV_OpenGlApiCallError, "Can't Activate The GL Rendering Context" );

        glViewport(0, 0, window->widget->allocation.width, window->widget->allocation.height);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (window->glDrawCallback)
            window->glDrawCallback(window->glDrawData);

        CV_CheckGlError();

        if (gdk_gl_drawable_is_double_buffered (gldrawable))
            gdk_gl_drawable_swap_buffers(gldrawable);
        else
            glFlush();

        gdk_gl_drawable_gl_end(gldrawable);

        __END__;
    }
}

#endif // HAVE_OPENGL


static gboolean cvImageWidget_expose(GtkWidget* widget, GdkEventExpose* event, gpointer data)
{
#ifdef HAVE_OPENGL
    CvWindow* window = (CvWindow*)data;

    if (window->useGl)
    {
        drawGl(window);
        return TRUE;
    }
#endif

  CvImageWidget *image_widget;

  g_return_val_if_fail (widget != NULL, FALSE);
  g_return_val_if_fail (CV_IS_IMAGE_WIDGET (widget), FALSE);
  g_return_val_if_fail (event != NULL, FALSE);

  if (event->count > 0)
    return FALSE;

  image_widget = CV_IMAGE_WIDGET (widget);

  gdk_window_clear_area (widget->window,
                         0, 0,
                         widget->allocation.width,
                         widget->allocation.height);
  if( image_widget->scaled_image ){
      // center image in available region
      int x0 = (widget->allocation.width - image_widget->scaled_image->cols)/2;
      int y0 = (widget->allocation.height - image_widget->scaled_image->rows)/2;

      gdk_draw_rgb_image( widget->window, widget->style->fg_gc[GTK_STATE_NORMAL],
          x0, y0, MIN(image_widget->scaled_image->cols, widget->allocation.width),
          MIN(image_widget->scaled_image->rows, widget->allocation.height),
          GDK_RGB_DITHER_MAX, image_widget->scaled_image->data.ptr, image_widget->scaled_image->step );
  }
  else if( image_widget->original_image ){
      gdk_draw_rgb_image( widget->window, widget->style->fg_gc[GTK_STATE_NORMAL],
          0, 0,
          MIN(image_widget->original_image->cols, widget->allocation.width),
           MIN(image_widget->original_image->rows, widget->allocation.height),
          GDK_RGB_DITHER_MAX, image_widget->original_image->data.ptr, image_widget->original_image->step );
  }
  return TRUE;
}

CV_IMPL int cvNamedWindow( const char* name, int flags )
{
    int result = 0;
    CV_FUNCNAME( "cvNamedWindow" );

    __BEGIN__;

    CvWindow* window;
    int len;

    cvInitSystem(1,(char**)&name);
    if( !name )
        CV_ERROR( CV_StsNullPtr, "NULL name string" );

    // Check the name in the storage
    if( icvFindWindowByName( name ) != 0 )
    {
        result = 1;
        EXIT;
    }

    len = strlen(name);
    CV_CALL( window = (CvWindow*)cvAlloc(sizeof(CvWindow) + len + 1));
    memset( window, 0, sizeof(*window));
    window->name = (char*)(window + 1);
    memcpy( window->name, name, len + 1 );
    window->flags = flags;
    window->signature = CV_WINDOW_MAGIC_VAL;
    window->last_key = 0;
    window->on_mouse = 0;
    window->on_mouse_param = 0;
    memset( &window->toolbar, 0, sizeof(window->toolbar));
    window->next = hg_windows;
    window->prev = 0;
    window->status = CV_WINDOW_NORMAL;//YV

    CV_LOCK_MUTEX();

    window->frame = gtk_window_new( GTK_WINDOW_TOPLEVEL );

    window->paned = gtk_vbox_new( FALSE, 0 );
    window->widget = cvImageWidgetNew( flags );
    gtk_box_pack_end( GTK_BOX(window->paned), window->widget, TRUE, TRUE, 0 );
    gtk_widget_show( window->widget );
    gtk_container_add( GTK_CONTAINER(window->frame), window->paned );
    gtk_widget_show( window->paned );

#ifndef HAVE_OPENGL
    if (flags & CV_WINDOW_OPENGL)
        CV_ERROR( CV_OpenGlNotSupported, "Library was built without OpenGL support" );
#else
    if (flags & CV_WINDOW_OPENGL)
        createGlContext(window);

    window->glDrawCallback = 0;
    window->glDrawData = 0;

    window->glCleanCallback = 0;
    window->glCleanData = 0;
#endif

    //
    // configure event handlers
    // TODO -- move this to CvImageWidget ?
    gtk_signal_connect( GTK_OBJECT(window->frame), "key-press-event",
                        GTK_SIGNAL_FUNC(icvOnKeyPress), window );
    gtk_signal_connect( GTK_OBJECT(window->widget), "button-press-event",
                        GTK_SIGNAL_FUNC(icvOnMouse), window );
    gtk_signal_connect( GTK_OBJECT(window->widget), "button-release-event",
                        GTK_SIGNAL_FUNC(icvOnMouse), window );
    gtk_signal_connect( GTK_OBJECT(window->widget), "motion-notify-event",
                        GTK_SIGNAL_FUNC(icvOnMouse), window );
    gtk_signal_connect( GTK_OBJECT(window->frame), "delete-event",
                        GTK_SIGNAL_FUNC(icvOnClose), window );
    gtk_signal_connect( GTK_OBJECT(window->widget), "expose-event",
                        GTK_SIGNAL_FUNC(cvImageWidget_expose), window );

    gtk_widget_add_events (window->widget, GDK_BUTTON_RELEASE_MASK | GDK_BUTTON_PRESS_MASK | GDK_POINTER_MOTION_MASK) ;

    gtk_widget_show( window->frame );
    gtk_window_set_title( GTK_WINDOW(window->frame), name );

    if( hg_windows )
        hg_windows->prev = window;
    hg_windows = window;

    gtk_window_set_resizable( GTK_WINDOW(window->frame), (flags & CV_WINDOW_AUTOSIZE) == 0 );


    // allow window to be resized
    if( (flags & CV_WINDOW_AUTOSIZE)==0 ){
        GdkGeometry geometry;
        geometry.min_width = 50;
        geometry.min_height = 50;
        gtk_window_set_geometry_hints( GTK_WINDOW( window->frame ), GTK_WIDGET( window->widget ),
            &geometry, (GdkWindowHints) (GDK_HINT_MIN_SIZE));
    }

    CV_UNLOCK_MUTEX();

#ifdef HAVE_OPENGL
    if (window->useGl)
        cvSetOpenGlContext(name);
#endif

    result = 1;
    __END__;

    return result;
}


#ifdef HAVE_OPENGL

CV_IMPL void cvSetOpenGlContext(const char* name)
{
    CvWindow* window;
    GdkGLContext* glcontext;
    GdkGLDrawable* gldrawable;

    CV_FUNCNAME( "cvSetOpenGlContext" );

    __BEGIN__;

    if(!name)
        CV_ERROR( CV_StsNullPtr, "NULL name string" );

    window = icvFindWindowByName( name );
    if (!window)
        CV_ERROR( CV_StsNullPtr, "NULL window" );

    if (!window->useGl)
        CV_ERROR( CV_OpenGlNotSupported, "Window doesn't support OpenGL" );

    glcontext = gtk_widget_get_gl_context(window->widget);
    gldrawable = gtk_widget_get_gl_drawable(window->widget);

    if (!gdk_gl_drawable_make_current(gldrawable, glcontext))
        CV_ERROR( CV_OpenGlApiCallError, "Can't Activate The GL Rendering Context" );

    __END__;
}

CV_IMPL void cvUpdateWindow(const char* name)
{
    CV_FUNCNAME( "cvUpdateWindow" );

    __BEGIN__;

    CvWindow* window;

    if (!name)
        CV_ERROR( CV_StsNullPtr, "NULL name string" );

    window = icvFindWindowByName( name );
    if (!window)
        EXIT;

    // window does not refresh without this
    gtk_widget_queue_draw( GTK_WIDGET(window->widget) );

    __END__;
}

CV_IMPL void cvSetOpenGlDrawCallback(const char* name, CvOpenGlDrawCallback callback, void* userdata)
{
    CvWindow* window;

    CV_FUNCNAME( "cvCreateOpenGLCallback" );

    __BEGIN__;

    if(!name)
        CV_ERROR( CV_StsNullPtr, "NULL name string" );

    window = icvFindWindowByName( name );
    if( !window )
        EXIT;

    if (!window->useGl)
        CV_ERROR( CV_OpenGlNotSupported, "Window was created without OpenGL context" );

    window->glDrawCallback = callback;
    window->glDrawData = userdata;

    __END__;
}

void icvSetOpenGlCleanCallback(const char* name, CvOpenGlCleanCallback callback, void* userdata)
{
    CvWindow* window;
    GdkGLContext* glcontext;
    GdkGLDrawable* gldrawable;

    CV_FUNCNAME( "icvSetOpenGlCleanCallback" );

    __BEGIN__;

    if (!name)
        CV_ERROR(CV_StsNullPtr, "NULL name string");

    window = icvFindWindowByName(name);
    if (!window)
        EXIT;

    if (!window->useGl)
        CV_ERROR( CV_OpenGlNotSupported, "Window doesn't support OpenGL" );

    glcontext = gtk_widget_get_gl_context(window->widget);
    gldrawable = gtk_widget_get_gl_drawable(window->widget);

    gdk_gl_drawable_make_current(gldrawable, glcontext);

    if (window->glCleanCallback)
        window->glCleanCallback(window->glCleanData);

    window->glCleanCallback = callback;
    window->glCleanData = userdata;

    __END__;
}

#endif // HAVE_OPENGL




static void icvDeleteWindow( CvWindow* window )
{
    CvTrackbar* trackbar;

#ifdef HAVE_OPENGL
    if (window->useGl)
    {
        GdkGLContext* glcontext = gtk_widget_get_gl_context(window->widget);
        GdkGLDrawable* gldrawable = gtk_widget_get_gl_drawable(window->widget);

        gdk_gl_drawable_make_current(gldrawable, glcontext);

        if (window->glCleanCallback)
        {
            window->glCleanCallback(window->glCleanData);
            window->glCleanCallback = 0;
            window->glCleanData = 0;
        }

        releaseGlContext(window);
    }
#endif

    if( window->prev )
        window->prev->next = window->next;
    else
        hg_windows = window->next;

    if( window->next )
        window->next->prev = window->prev;

    window->prev = window->next = 0;

    gtk_widget_destroy( window->frame );

    for( trackbar = window->toolbar.first; trackbar != 0; )
    {
        CvTrackbar* next = trackbar->next;
        cvFree( &trackbar );
        trackbar = next;
    }

    cvFree( &window );
#ifdef HAVE_GTHREAD
    // if last window, send key press signal
    // to jump out of any waiting cvWaitKey's
    if(hg_windows==0 && thread_started){
        g_cond_broadcast(cond_have_key);
    }
#endif
}


CV_IMPL void cvDestroyWindow( const char* name )
{
    CV_FUNCNAME( "cvDestroyWindow" );

    __BEGIN__;

    CvWindow* window;

    if(!name)
        CV_ERROR( CV_StsNullPtr, "NULL name string" );

    window = icvFindWindowByName( name );
    if( !window )
        EXIT;

    // note that it is possible for the update thread to run this function
    // if there is a call to cvShowImage in a mouse callback
    // (this would produce a deadlock on window_mutex)
    CV_LOCK_MUTEX();

    icvDeleteWindow( window );

    CV_UNLOCK_MUTEX();

    __END__;
}


CV_IMPL void
cvDestroyAllWindows( void )
{
    CV_LOCK_MUTEX();

    while( hg_windows )
    {
        CvWindow* window = hg_windows;
        icvDeleteWindow( window );
    }
    CV_UNLOCK_MUTEX();
}

// CvSize icvCalcOptimalWindowSize( CvWindow * window, CvSize new_image_size){
//     CvSize window_size;
//     GtkWidget * toplevel = gtk_widget_get_toplevel( window->frame );
//     gdk_drawable_get_size( GDK_DRAWABLE(toplevel->window),
//             &window_size.width, &window_size.height );

//     window_size.width = window_size.width + new_image_size.width - window->widget->allocation.width;
//     window_size.height = window_size.height + new_image_size.height - window->widget->allocation.height;

//     return window_size;
// }

CV_IMPL void
cvShowImage( const char* name, const CvArr* arr )
{
    CV_FUNCNAME( "cvShowImage" );

    __BEGIN__;

    CvWindow* window;

    if( !name )
        CV_ERROR( CV_StsNullPtr, "NULL name" );

    CV_LOCK_MUTEX();

    window = icvFindWindowByName(name);
    if(!window)
    {
        cvNamedWindow(name, 1);
        window = icvFindWindowByName(name);
    }

    if( window && arr )
    {
    #ifdef HAVE_OPENGL
        if (window->useGl)
        {
            CvMat stub;
            CvMat* mat = cvGetMat(arr, &stub);
            cv::Mat im(mat);
            cv::imshow(name, im);
            return;
        }
    #endif

        CvImageWidget * image_widget = CV_IMAGE_WIDGET( window->widget );
        cvImageWidgetSetImage( image_widget, arr );
    }

    CV_UNLOCK_MUTEX();

    __END__;
}

CV_IMPL void cvResizeWindow(const char* name, int width, int height )
{
    CV_FUNCNAME( "cvResizeWindow" );

    __BEGIN__;

    CvWindow* window;
    CvImageWidget * image_widget;

    if( !name )
        CV_ERROR( CV_StsNullPtr, "NULL name" );

    window = icvFindWindowByName(name);
    if(!window)
        EXIT;

    image_widget = CV_IMAGE_WIDGET( window->widget );
    //if(image_widget->flags & CV_WINDOW_AUTOSIZE)
        //EXIT;

    CV_LOCK_MUTEX();

    gtk_window_set_resizable( GTK_WINDOW(window->frame), 1 );
    gtk_window_resize( GTK_WINDOW(window->frame), width, height );

    // disable initial resize since presumably user wants to keep
    // this window size
    image_widget->flags &= ~CV_WINDOW_NO_IMAGE;

    CV_UNLOCK_MUTEX();

    __END__;
}


CV_IMPL void cvMoveWindow( const char* name, int x, int y )
{
    CV_FUNCNAME( "cvMoveWindow" );

    __BEGIN__;

    CvWindow* window;

    if( !name )
        CV_ERROR( CV_StsNullPtr, "NULL name" );

    window = icvFindWindowByName(name);
    if(!window)
        EXIT;

    CV_LOCK_MUTEX();

    gtk_window_move( GTK_WINDOW(window->frame), x, y );

    CV_UNLOCK_MUTEX();

    __END__;
}


static CvTrackbar*
icvFindTrackbarByName( const CvWindow* window, const char* name )
{
    CvTrackbar* trackbar = window->toolbar.first;

    for( ; trackbar != 0 && strcmp( trackbar->name, name ) != 0; trackbar = trackbar->next )
        ;

    return trackbar;
}

static int
icvCreateTrackbar( const char* trackbar_name, const char* window_name,
                   int* val, int count, CvTrackbarCallback on_notify,
                   CvTrackbarCallback2 on_notify2, void* userdata )
{
    int result = 0;

    CV_FUNCNAME( "icvCreateTrackbar" );

    __BEGIN__;

    /*char slider_name[32];*/
    CvWindow* window = 0;
    CvTrackbar* trackbar = 0;

    if( !window_name || !trackbar_name )
        CV_ERROR( CV_StsNullPtr, "NULL window or trackbar name" );

    if( count <= 0 )
        CV_ERROR( CV_StsOutOfRange, "Bad trackbar maximal value" );

    window = icvFindWindowByName(window_name);
    if( !window )
        EXIT;

    trackbar = icvFindTrackbarByName(window,trackbar_name);

    CV_LOCK_MUTEX();

    if( !trackbar )
    {
        int len = strlen(trackbar_name);
        trackbar = (CvTrackbar*)cvAlloc(sizeof(CvTrackbar) + len + 1);
        memset( trackbar, 0, sizeof(*trackbar));
        trackbar->signature = CV_TRACKBAR_MAGIC_VAL;
        trackbar->name = (char*)(trackbar+1);
        memcpy( trackbar->name, trackbar_name, len + 1 );
        trackbar->parent = window;
        trackbar->next = window->toolbar.first;
        window->toolbar.first = trackbar;

        GtkWidget* hscale_box = gtk_hbox_new( FALSE, 10 );
        GtkWidget* hscale_label = gtk_label_new( trackbar_name );
        GtkWidget* hscale = gtk_hscale_new_with_range( 0, count, 1 );
        gtk_range_set_update_policy( GTK_RANGE(hscale), GTK_UPDATE_CONTINUOUS );
        gtk_scale_set_digits( GTK_SCALE(hscale), 0 );
        //gtk_scale_set_value_pos( hscale, GTK_POS_TOP );
        gtk_scale_set_draw_value( GTK_SCALE(hscale), TRUE );

        trackbar->widget = hscale;
        gtk_box_pack_start( GTK_BOX(hscale_box), hscale_label, FALSE, FALSE, 5 );
        gtk_widget_show( hscale_label );
        gtk_box_pack_start( GTK_BOX(hscale_box), hscale, TRUE, TRUE, 5 );
        gtk_widget_show( hscale );
        gtk_box_pack_start( GTK_BOX(window->paned), hscale_box, FALSE, FALSE, 5 );
        gtk_widget_show( hscale_box );

    }

    if( val )
    {
        int value = *val;
        if( value < 0 )
            value = 0;
        if( value > count )
            value = count;
        gtk_range_set_value( GTK_RANGE(trackbar->widget), value );
        trackbar->pos = value;
        trackbar->data = val;
    }

    trackbar->maxval = count;
    trackbar->notify = on_notify;
    trackbar->notify2 = on_notify2;
    trackbar->userdata = userdata;
    gtk_signal_connect( GTK_OBJECT(trackbar->widget), "value-changed",
                        GTK_SIGNAL_FUNC(icvOnTrackbar), trackbar );

    // queue a widget resize to trigger a window resize to
    // compensate for the addition of trackbars
    gtk_widget_queue_resize( GTK_WIDGET(window->widget) );


    CV_UNLOCK_MUTEX();

    result = 1;

    __END__;

    return result;
}


CV_IMPL int
cvCreateTrackbar( const char* trackbar_name, const char* window_name,
                  int* val, int count, CvTrackbarCallback on_notify )
{
    return icvCreateTrackbar(trackbar_name, window_name, val, count,
                             on_notify, 0, 0);
}


CV_IMPL int
cvCreateTrackbar2( const char* trackbar_name, const char* window_name,
                   int* val, int count, CvTrackbarCallback2 on_notify2,
                   void* userdata )
{
    return icvCreateTrackbar(trackbar_name, window_name, val, count,
                             0, on_notify2, userdata);
}


CV_IMPL void
cvSetMouseCallback( const char* window_name, CvMouseCallback on_mouse, void* param )
{
    CV_FUNCNAME( "cvSetMouseCallback" );

    __BEGIN__;

    CvWindow* window = 0;

    if( !window_name )
        CV_ERROR( CV_StsNullPtr, "NULL window name" );

    window = icvFindWindowByName(window_name);
    if( !window )
        EXIT;

    window->on_mouse = on_mouse;
    window->on_mouse_param = param;

    __END__;
}


CV_IMPL int cvGetTrackbarPos( const char* trackbar_name, const char* window_name )
{
    int pos = -1;

    CV_FUNCNAME( "cvGetTrackbarPos" );

    __BEGIN__;

    CvWindow* window;
    CvTrackbar* trackbar = 0;

    if( trackbar_name == 0 || window_name == 0 )
        CV_ERROR( CV_StsNullPtr, "NULL trackbar or window name" );

    window = icvFindWindowByName( window_name );
    if( window )
        trackbar = icvFindTrackbarByName( window, trackbar_name );

    if( trackbar )
        pos = trackbar->pos;

    __END__;

    return pos;
}


CV_IMPL void cvSetTrackbarPos( const char* trackbar_name, const char* window_name, int pos )
{
    CV_FUNCNAME( "cvSetTrackbarPos" );

    __BEGIN__;

    CvWindow* window;
    CvTrackbar* trackbar = 0;

    if( trackbar_name == 0 || window_name == 0 )
        CV_ERROR( CV_StsNullPtr, "NULL trackbar or window name" );

    window = icvFindWindowByName( window_name );
    if( window )
        trackbar = icvFindTrackbarByName( window, trackbar_name );

    if( trackbar )
    {
        if( pos < 0 )
            pos = 0;

        if( pos > trackbar->maxval )
            pos = trackbar->maxval;
    }

    CV_LOCK_MUTEX();

    gtk_range_set_value( GTK_RANGE(trackbar->widget), pos );

    CV_UNLOCK_MUTEX();

    __END__;
}


CV_IMPL void* cvGetWindowHandle( const char* window_name )
{
    void* widget = 0;

    CV_FUNCNAME( "cvGetWindowHandle" );

    __BEGIN__;

    CvWindow* window;

    if( window_name == 0 )
        CV_ERROR( CV_StsNullPtr, "NULL window name" );

    window = icvFindWindowByName( window_name );
    if( window )
        widget = (void*)window->widget;

    __END__;

    return widget;
}


CV_IMPL const char* cvGetWindowName( void* window_handle )
{
    const char* window_name = "";

    CV_FUNCNAME( "cvGetWindowName" );

    __BEGIN__;

    CvWindow* window;

    if( window_handle == 0 )
        CV_ERROR( CV_StsNullPtr, "NULL window" );

    window = icvWindowByWidget( (GtkWidget*)window_handle );
    if( window )
        window_name = window->name;

    __END__;

    return window_name;
}

static gboolean icvOnKeyPress( GtkWidget * /*widget*/,
                GdkEventKey* event, gpointer /*user_data*/ )
{
    int code = 0;

    switch( event->keyval )
    {
    case GDK_Escape:
        code = 27;
        break;
    case GDK_Return:
    case GDK_Linefeed:
        code = '\n';
        break;
    case GDK_Tab:
        code = '\t';
    break;
    default:
        code = event->keyval;
    }

    code |= event->state << 16;

#ifdef HAVE_GTHREAD
    if(thread_started) g_mutex_lock(last_key_mutex);
#endif

    last_key = code;

#ifdef HAVE_GTHREAD
    if(thread_started){
        // signal any waiting threads
        g_cond_broadcast(cond_have_key);
        g_mutex_unlock(last_key_mutex);
    }
#endif

    return FALSE;
}


static void icvOnTrackbar( GtkWidget* widget, gpointer user_data )
{
    int pos = cvRound( gtk_range_get_value(GTK_RANGE(widget)));
    CvTrackbar* trackbar = (CvTrackbar*)user_data;

    if( trackbar && trackbar->signature == CV_TRACKBAR_MAGIC_VAL &&
        trackbar->widget == widget )
    {
        trackbar->pos = pos;
        if( trackbar->data )
            *trackbar->data = pos;
        if( trackbar->notify2 )
            trackbar->notify2(pos, trackbar->userdata);
        else if( trackbar->notify )
            trackbar->notify(pos);
    }
}

static gboolean icvOnClose( GtkWidget* widget, GdkEvent* /*event*/, gpointer user_data )
{
    CvWindow* window = (CvWindow*)user_data;
    if( window->signature == CV_WINDOW_MAGIC_VAL &&
        window->frame == widget )
    {
        icvDeleteWindow(window);
    }
    return TRUE;
}


static gboolean icvOnMouse( GtkWidget *widget, GdkEvent *event, gpointer user_data )
{
    // TODO move this logic to CvImageWidget
    CvWindow* window = (CvWindow*)user_data;
    CvPoint2D32f pt32f = {-1., -1.};
    CvPoint pt = {-1,-1};
    int cv_event = -1, state = 0;
    CvImageWidget * image_widget = CV_IMAGE_WIDGET( widget );

    if( window->signature != CV_WINDOW_MAGIC_VAL ||
        window->widget != widget || !window->widget ||
        !window->on_mouse /*|| !image_widget->original_image*/)
        return FALSE;

    if( event->type == GDK_MOTION_NOTIFY )
    {
        GdkEventMotion* event_motion = (GdkEventMotion*)event;

        cv_event = CV_EVENT_MOUSEMOVE;
        pt32f.x = cvRound(event_motion->x);
        pt32f.y = cvRound(event_motion->y);
        state = event_motion->state;
    }
    else if( event->type == GDK_BUTTON_PRESS ||
             event->type == GDK_BUTTON_RELEASE ||
             event->type == GDK_2BUTTON_PRESS )
    {
        GdkEventButton* event_button = (GdkEventButton*)event;
        pt32f.x = cvRound(event_button->x);
        pt32f.y = cvRound(event_button->y);


        if( event_button->type == GDK_BUTTON_PRESS )
        {
            cv_event = event_button->button == 1 ? CV_EVENT_LBUTTONDOWN :
                       event_button->button == 2 ? CV_EVENT_MBUTTONDOWN :
                       event_button->button == 3 ? CV_EVENT_RBUTTONDOWN : 0;
        }
        else if( event_button->type == GDK_BUTTON_RELEASE )
        {
            cv_event = event_button->button == 1 ? CV_EVENT_LBUTTONUP :
                       event_button->button == 2 ? CV_EVENT_MBUTTONUP :
                       event_button->button == 3 ? CV_EVENT_RBUTTONUP : 0;
        }
        else if( event_button->type == GDK_2BUTTON_PRESS )
        {
            cv_event = event_button->button == 1 ? CV_EVENT_LBUTTONDBLCLK :
                       event_button->button == 2 ? CV_EVENT_MBUTTONDBLCLK :
                       event_button->button == 3 ? CV_EVENT_RBUTTONDBLCLK : 0;
        }
        state = event_button->state;
    }

    if( cv_event >= 0 ){
        // scale point if image is scaled
        if( (image_widget->flags & CV_WINDOW_AUTOSIZE)==0 &&
             image_widget->original_image &&
             image_widget->scaled_image ){
            // image origin is not necessarily at (0,0)
            int x0 = (widget->allocation.width - image_widget->scaled_image->cols)/2;
            int y0 = (widget->allocation.height - image_widget->scaled_image->rows)/2;
            pt.x = cvRound( ((pt32f.x-x0)*image_widget->original_image->cols)/
                                            image_widget->scaled_image->cols );
            pt.y = cvRound( ((pt32f.y-y0)*image_widget->original_image->rows)/
                                            image_widget->scaled_image->rows );
        }
        else{
            pt = cvPointFrom32f( pt32f );
        }

//        if((unsigned)pt.x < (unsigned)(image_widget->original_image->width) &&
//           (unsigned)pt.y < (unsigned)(image_widget->original_image->height) )
        {
            int flags = (state & GDK_SHIFT_MASK ? CV_EVENT_FLAG_SHIFTKEY : 0) |
                (state & GDK_CONTROL_MASK ? CV_EVENT_FLAG_CTRLKEY : 0) |
                (state & (GDK_MOD1_MASK|GDK_MOD2_MASK) ? CV_EVENT_FLAG_ALTKEY : 0) |
                (state & GDK_BUTTON1_MASK ? CV_EVENT_FLAG_LBUTTON : 0) |
                (state & GDK_BUTTON2_MASK ? CV_EVENT_FLAG_MBUTTON : 0) |
                (state & GDK_BUTTON3_MASK ? CV_EVENT_FLAG_RBUTTON : 0);
            window->on_mouse( cv_event, pt.x, pt.y, flags, window->on_mouse_param );
        }
    }

        return FALSE;
    }


static gboolean icvAlarm( gpointer user_data )
{
    *(int*)user_data = 1;
    return FALSE;
}


CV_IMPL int cvWaitKey( int delay )
{
#ifdef HAVE_GTHREAD
    if(thread_started && g_thread_self()!=window_thread){
        gboolean expired;
        int my_last_key;

        // wait for signal or timeout if delay > 0
        if(delay>0){
            GTimeVal timer;
            g_get_current_time(&timer);
            g_time_val_add(&timer, delay*1000);
            expired = !g_cond_timed_wait(cond_have_key, last_key_mutex, &timer);
        }
        else{
            g_cond_wait(cond_have_key, last_key_mutex);
            expired=false;
        }
        my_last_key = last_key;
        g_mutex_unlock(last_key_mutex);
        if(expired || hg_windows==0){
            return -1;
        }
        return my_last_key;
    }
    else{
#endif
        int expired = 0;
        guint timer = 0;
        if( delay > 0 )
            timer = g_timeout_add( delay, icvAlarm, &expired );
        last_key = -1;
        while( gtk_main_iteration_do(TRUE) && last_key < 0 && !expired && hg_windows != 0 )
            ;

        if( delay > 0 && !expired )
            g_source_remove(timer);
#ifdef HAVE_GTHREAD
    }
#endif
    return last_key;
}


#endif  // HAVE_GTK
#endif  // WIN32

/* End of file. */
