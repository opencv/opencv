/* GStreamer
 * Copyright (C) 2007 David Schleef <ds@schleef.org>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */
#if 1
#include "precomp.hpp"
#include <gst/gst.h>
#include <gst/base/gstbasesink.h>
#include <gst/gstbuffer.h>

#include <string.h>

#include "gstappsink.h"

GST_DEBUG_CATEGORY (app_sink_debug);
#define GST_CAT_DEFAULT app_sink_debug

static const GstElementDetails app_sink_details =
GST_ELEMENT_DETAILS ((gchar*)"AppSink",
    (gchar*)"Generic/Sink",
    (gchar*)"Allow the application to get access to raw buffer",
    (gchar*)"David Schleef <ds@schleef.org>, Wim Taymans <wim.taymans@gmail.com");

enum
{
  /* signals */
  SIGNAL_EOS,
  SIGNAL_NEW_PREROLL,
  SIGNAL_NEW_BUFFER,

  /* acions */
  SIGNAL_PULL_PREROLL,
  SIGNAL_PULL_BUFFER,

  LAST_SIGNAL
};

enum
{
  PROP_0,
  PROP_CAPS,
  PROP_EOS
};

static GstStaticPadTemplate gst_app_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS_ANY);

static void gst_app_sink_dispose (GObject * object);
static void gst_app_sink_finalize (GObject * object);

static void gst_app_sink_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_app_sink_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_app_sink_start (GstBaseSink * psink);
static gboolean gst_app_sink_stop (GstBaseSink * psink);
static gboolean gst_app_sink_event (GstBaseSink * sink, GstEvent * event);
static GstFlowReturn gst_app_sink_preroll (GstBaseSink * psink,
    GstBuffer * buffer);
static GstFlowReturn gst_app_sink_render (GstBaseSink * psink,
    GstBuffer * buffer);
static GstCaps *gst_app_sink_getcaps (GstBaseSink * psink);

static guint gst_app_sink_signals[LAST_SIGNAL] = { 0 };

GST_BOILERPLATE (GstAppSink, gst_app_sink, GstBaseSink, GST_TYPE_BASE_SINK);

static gboolean
appsink_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (app_sink_debug, "opencv-appsink", 0, "Application sink");

  if (!gst_element_register (plugin, "opencv-appsink", GST_RANK_PRIMARY,
          gst_app_sink_get_type ()))
    return FALSE;

  return TRUE;
}

#undef PACKAGE
#define PACKAGE "highgui"
GST_PLUGIN_DEFINE_STATIC (GST_VERSION_MAJOR, GST_VERSION_MINOR,
	"opencv-appsink", "Element application sink",
	appsink_plugin_init, "0.1", "LGPL", "OpenCV's internal copy of gstappsink", "OpenCV")

void
gst_app_marshal_OBJECT__VOID (GClosure * closure,
    GValue * return_value,
    guint n_param_values,
    const GValue * param_values,
    gpointer invocation_hint, gpointer marshal_data)
{
  typedef GstBuffer *(*GMarshalFunc_OBJECT__VOID) (gpointer data1,
      gpointer data2);
  register GMarshalFunc_OBJECT__VOID callback;
  register GCClosure *cc = (GCClosure *) closure;
  register gpointer data1, data2;
  GstBuffer *v_return;

  //printf("appmarshalobject\n");

  g_return_if_fail (return_value != NULL);
  g_return_if_fail (n_param_values == 1);

  if (G_CCLOSURE_SWAP_DATA (closure)) {
    data1 = closure->data;
    data2 = g_value_peek_pointer (param_values + 0);
  } else {
    data1 = g_value_peek_pointer (param_values + 0);
    data2 = closure->data;
  }
  callback =
      (GMarshalFunc_OBJECT__VOID) (marshal_data ? marshal_data : cc->callback);

  v_return = callback (data1, data2);

  gst_value_take_buffer (return_value, v_return);
}

static void
gst_app_sink_base_init (gpointer g_class)
{
  GstElementClass *element_class = GST_ELEMENT_CLASS (g_class);

  //printf("appsinkbaseinit\n");

  GST_DEBUG_CATEGORY_INIT (app_sink_debug, "appsink", 0, "appsink element");

  gst_element_class_set_details (element_class, &app_sink_details);

  gst_element_class_add_pad_template (element_class,
      gst_static_pad_template_get (&gst_app_sink_template));
}

static void
gst_app_sink_class_init (GstAppSinkClass * klass)
{
  GObjectClass *gobject_class = (GObjectClass *) klass;
  GstBaseSinkClass *basesink_class = (GstBaseSinkClass *) klass;

  //printf("appsinkclassinit\n");

  gobject_class->dispose = gst_app_sink_dispose;
  gobject_class->finalize = gst_app_sink_finalize;

  gobject_class->set_property = gst_app_sink_set_property;
  gobject_class->get_property = gst_app_sink_get_property;

  g_object_class_install_property (gobject_class, PROP_CAPS,
      g_param_spec_boxed ("caps", "Caps",
          "The caps of the sink pad", GST_TYPE_CAPS, (GParamFlags)G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_EOS,
      g_param_spec_boolean ("eos", "EOS",
          "Check if the sink is EOS", TRUE, G_PARAM_READABLE));

  /**
   * GstAppSink::eos:
   * @appsink: the appsink element that emitted the signal
   *
   * Signal that the end-of-stream has been reached.
   */
  gst_app_sink_signals[SIGNAL_EOS] =
      g_signal_new ("eos", G_TYPE_FROM_CLASS (klass), G_SIGNAL_RUN_LAST,
      G_STRUCT_OFFSET (GstAppSinkClass, eos),
      NULL, NULL, g_cclosure_marshal_VOID__VOID, G_TYPE_NONE, 0, G_TYPE_NONE);
  /**
   * GstAppSink::new-preroll:
   * @appsink: the appsink element that emitted the signal
   * @buffer: the buffer that caused the preroll
   *
   * Signal that a new preroll buffer is available.
   */
  gst_app_sink_signals[SIGNAL_NEW_PREROLL] =
      g_signal_new ("new-preroll", G_TYPE_FROM_CLASS (klass), G_SIGNAL_RUN_LAST,
      G_STRUCT_OFFSET (GstAppSinkClass, new_preroll),
      NULL, NULL, g_cclosure_marshal_VOID__OBJECT, G_TYPE_NONE, 1,
      GST_TYPE_BUFFER);
  /**
   * GstAppSink::new-buffer:
   * @appsink: the appsink element that emitted the signal
   * @buffer: the buffer that is available
   *
   * Signal that a new buffer is available.
   */
  gst_app_sink_signals[SIGNAL_NEW_BUFFER] =
      g_signal_new ("new-buffer", G_TYPE_FROM_CLASS (klass), G_SIGNAL_RUN_LAST,
      G_STRUCT_OFFSET (GstAppSinkClass, new_buffer),
      NULL, NULL, g_cclosure_marshal_VOID__UINT, G_TYPE_NONE, 1,
      G_TYPE_UINT);

  /**
   * GstAppSink::pull-preroll:
   * @appsink: the appsink element to emit this signal on
   *
   * Get the last preroll buffer on @appsink.
   */
  gst_app_sink_signals[SIGNAL_PULL_PREROLL] =
      g_signal_new ("pull-preroll", G_TYPE_FROM_CLASS (klass),
      G_SIGNAL_RUN_LAST, G_STRUCT_OFFSET (GstAppSinkClass, pull_preroll), NULL,
      NULL, gst_app_marshal_OBJECT__VOID, GST_TYPE_BUFFER, 0, G_TYPE_NONE);
  /**
   * GstAppSink::pull-buffer:
   * @appsink: the appsink element to emit this signal on
   *
   * Get the next buffer buffer on @appsink.
   */
  gst_app_sink_signals[SIGNAL_PULL_PREROLL] =
      g_signal_new ("pull-buffer", G_TYPE_FROM_CLASS (klass), G_SIGNAL_RUN_LAST,
      G_STRUCT_OFFSET (GstAppSinkClass, pull_buffer),
      NULL, NULL, gst_app_marshal_OBJECT__VOID, GST_TYPE_BUFFER, 0,
      G_TYPE_NONE);

  basesink_class->start = gst_app_sink_start;
  basesink_class->stop = gst_app_sink_stop;
  basesink_class->event = gst_app_sink_event;
  basesink_class->preroll = gst_app_sink_preroll;
  basesink_class->render = gst_app_sink_render;
  basesink_class->get_caps = gst_app_sink_getcaps;

  klass->pull_preroll = gst_app_sink_pull_preroll;
  klass->pull_buffer = gst_app_sink_pull_buffer;
}

static void
gst_app_sink_init (GstAppSink * appsink, GstAppSinkClass * klass)
{
  appsink->mutex = g_mutex_new ();
  appsink->cond = g_cond_new ();
  appsink->queue = g_queue_new ();
}

static void
gst_app_sink_dispose (GObject * obj)
{
  GstAppSink *appsink = GST_APP_SINK (obj);
  GstBuffer *buffer;

  //printf("appsinkdispose\n");

  if (appsink->caps) {
    gst_caps_unref (appsink->caps);
    appsink->caps = NULL;
  }
  if (appsink->preroll) {
    gst_buffer_unref (appsink->preroll);
    appsink->preroll = NULL;
  }
  g_mutex_lock (appsink->mutex);
  while ((buffer = (GstBuffer*)g_queue_pop_head (appsink->queue)))
    gst_buffer_unref (buffer);
  g_mutex_unlock (appsink->mutex);

  G_OBJECT_CLASS (parent_class)->dispose (obj);
}

static void
gst_app_sink_finalize (GObject * obj)
{
  GstAppSink *appsink = GST_APP_SINK (obj);

  g_mutex_free (appsink->mutex);
  g_cond_free (appsink->cond);
  g_queue_free (appsink->queue);

  G_OBJECT_CLASS (parent_class)->finalize (obj);
}

static void
gst_app_sink_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstAppSink *appsink = GST_APP_SINK (object);

  //printf("appsinksetproperty\n");

  switch (prop_id) {
    case PROP_CAPS:
      gst_app_sink_set_caps (appsink, gst_value_get_caps (value));
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_app_sink_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstAppSink *appsink = GST_APP_SINK (object);

  //printf("appsinkgetproperty\n");

  switch (prop_id) {
    case PROP_CAPS:
    {
      GstCaps *caps;

      caps = gst_app_sink_get_caps (appsink);
      gst_value_set_caps (value, caps);
      if (caps)
        gst_caps_unref (caps);
      break;
    }
    case PROP_EOS:
      g_value_set_boolean (value, gst_app_sink_is_eos (appsink));
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

static void
gst_app_sink_flush_unlocked (GstAppSink * appsink)
{
  GstBuffer *buffer;

  //printf("appsinkflushunlocked\n");

  GST_DEBUG_OBJECT (appsink, "flushing appsink");
  appsink->is_eos = FALSE;
  gst_buffer_replace (&appsink->preroll, NULL);
  while ((buffer = (GstBuffer*)g_queue_pop_head (appsink->queue)))
    gst_buffer_unref (buffer);
  g_cond_signal (appsink->cond);
}

static gboolean
gst_app_sink_start (GstBaseSink * psink)
{
  GstAppSink *appsink = GST_APP_SINK (psink);

  //printf("appsinkstart\n");

  g_mutex_lock (appsink->mutex);
  appsink->is_eos = FALSE;
  appsink->started = TRUE;
  GST_DEBUG_OBJECT (appsink, "starting");
  g_mutex_unlock (appsink->mutex);

  return TRUE;
}

static gboolean
gst_app_sink_stop (GstBaseSink * psink)
{
  GstAppSink *appsink = GST_APP_SINK (psink);

  //printf("appsinkstop\n");

  g_mutex_lock (appsink->mutex);
  GST_DEBUG_OBJECT (appsink, "stopping");
  appsink->started = FALSE;
  gst_app_sink_flush_unlocked (appsink);
  g_mutex_unlock (appsink->mutex);

  return TRUE;
}

static gboolean
gst_app_sink_event (GstBaseSink * sink, GstEvent * event)
{
  GstAppSink *appsink = GST_APP_SINK (sink);

  //printf("appsinkevent\n");

  switch (event->type) {
    case GST_EVENT_EOS:
      g_mutex_lock (appsink->mutex);
      GST_DEBUG_OBJECT (appsink, "receiving EOS");
      appsink->is_eos = TRUE;
      g_cond_signal (appsink->cond);
      g_mutex_unlock (appsink->mutex);
      break;
    case GST_EVENT_FLUSH_START:
      break;
    case GST_EVENT_FLUSH_STOP:
      g_mutex_lock (appsink->mutex);
      GST_DEBUG_OBJECT (appsink, "received FLUSH_STOP");
      gst_app_sink_flush_unlocked (appsink);
      g_mutex_unlock (appsink->mutex);
      break;
    default:
      break;
  }
  return TRUE;
}

static GstFlowReturn
gst_app_sink_preroll (GstBaseSink * psink, GstBuffer * buffer)
{
  GstAppSink *appsink = GST_APP_SINK (psink);

  //printf("appsinkpreroll\n");

  g_mutex_lock (appsink->mutex);
  GST_DEBUG_OBJECT (appsink, "setting preroll buffer %p", buffer);
  gst_buffer_replace (&appsink->preroll, buffer);
  g_cond_signal (appsink->cond);
  g_mutex_unlock (appsink->mutex);

  g_signal_emit(psink, gst_app_sink_signals[SIGNAL_NEW_PREROLL], 0, buffer);

  return GST_FLOW_OK;
}

static GstFlowReturn
gst_app_sink_render (GstBaseSink * psink, GstBuffer * buffer)
{
  GstAppSink *appsink = GST_APP_SINK (psink);

  g_mutex_lock (appsink->mutex);
  GST_DEBUG_OBJECT (appsink, "pushing render buffer %p on queue", buffer);
  g_queue_push_tail (appsink->queue, gst_buffer_ref (buffer));
  g_cond_signal (appsink->cond);
//  printf("appsinkrender, have %d buffers\n", g_queue_get_length(appsink->queue));
  g_mutex_unlock (appsink->mutex);
  g_signal_emit(psink, gst_app_sink_signals[SIGNAL_NEW_BUFFER], 0,
                g_queue_get_length(appsink->queue));

  return GST_FLOW_OK;
}

static GstCaps *
gst_app_sink_getcaps (GstBaseSink * psink)
{
  GstCaps *caps;

  //printf("appsinkgetcaps\n");

  GstAppSink *appsink = GST_APP_SINK (psink);

  GST_OBJECT_LOCK (appsink);
  if ((caps = appsink->caps))
    gst_caps_ref (caps);
  GST_DEBUG_OBJECT (appsink, "got caps %" GST_PTR_FORMAT, caps);
  GST_OBJECT_UNLOCK (appsink);

  return caps;
}

/* external API */

/**
 * gst_app_sink_set_caps:
 * @appsink: a #GstAppSink
 * @caps: caps to set
 *
 * Set the capabilities on the appsink element.  This function takes
 * a copy of the caps structure. After calling this method, the sink will only
 * accept caps that match @caps. If @caps is non-fixed, you must check the caps
 * on the buffers to get the actual used caps.
 */
void
gst_app_sink_set_caps (GstAppSink * appsink, const GstCaps * caps)
{
  GstCaps *old;

  g_return_if_fail (appsink != NULL);
  g_return_if_fail (GST_IS_APP_SINK (appsink));

  GST_OBJECT_LOCK (appsink);
  GST_DEBUG_OBJECT (appsink, "setting caps to %" GST_PTR_FORMAT, caps);
  old = appsink->caps;
  if (caps)
    appsink->caps = gst_caps_copy (caps);
  else
    appsink->caps = NULL;
  if (old)
    gst_caps_unref (old);
  GST_OBJECT_UNLOCK (appsink);
}

/**
 * gst_app_sink_get_caps:
 * @appsink: a #GstAppSink
 *
 * Get the configured caps on @appsink.
 *
 * Returns: the #GstCaps accepted by the sink. gst_caps_unref() after usage.
 */
GstCaps *
gst_app_sink_get_caps (GstAppSink * appsink)
{
  GstCaps *caps;

  g_return_val_if_fail (appsink != NULL, NULL);
  g_return_val_if_fail (GST_IS_APP_SINK (appsink), NULL);

  GST_OBJECT_LOCK (appsink);
  if ((caps = appsink->caps))
    gst_caps_ref (caps);
  GST_DEBUG_OBJECT (appsink, "getting caps of %" GST_PTR_FORMAT, caps);
  GST_OBJECT_UNLOCK (appsink);

  return caps;
}

/**
 * gst_app_sink_is_eos:
 * @appsink: a #GstAppSink
 *
 * Check if @appsink is EOS, which is when no more buffers can be pulled because
 * an EOS event was received.
 *
 * This function also returns %TRUE when the appsink is not in the PAUSED or
 * PLAYING state.
 *
 * Returns: %TRUE if no more buffers can be pulled and the appsink is EOS.
 */
gboolean
gst_app_sink_is_eos (GstAppSink * appsink)
{
  gboolean ret;

  g_return_val_if_fail (appsink != NULL, FALSE);
  g_return_val_if_fail (GST_IS_APP_SINK (appsink), FALSE);

  g_mutex_lock (appsink->mutex);
  if (!appsink->started)
    goto not_started;

  if (appsink->is_eos && g_queue_is_empty (appsink->queue)) {
    GST_DEBUG_OBJECT (appsink, "we are EOS and the queue is empty");
    ret = TRUE;
  } else {
    GST_DEBUG_OBJECT (appsink, "we are not yet EOS");
    ret = FALSE;
  }
  g_mutex_unlock (appsink->mutex);

  return ret;

not_started:
  {
    GST_DEBUG_OBJECT (appsink, "we are stopped, return TRUE");
    g_mutex_unlock (appsink->mutex);
    return TRUE;
  }
}

/**
 * gst_app_sink_pull_preroll:
 * @appsink: a #GstAppSink
 *
 * Get the last preroll buffer in @appsink. This was the buffer that caused the
 * appsink to preroll in the PAUSED state. This buffer can be pulled many times
 * and remains available to the application even after EOS.
 *
 * This function is typically used when dealing with a pipeline in the PAUSED
 * state. Calling this function after doing a seek will give the buffer right
 * after the seek position.
 *
 * Note that the preroll buffer will also be returned as the first buffer
 * when calling gst_app_sink_pull_buffer().
 *
 * If an EOS event was received before any buffers, this function returns
 * %NULL. Use gst_app_sink_is_eos () to check for the EOS condition.
 *
 * This function blocks until a preroll buffer or EOS is received or the appsink
 * element is set to the READY/NULL state.
 *
 * Returns: a #GstBuffer or NULL when the appsink is stopped or EOS.
 */
GstBuffer *
gst_app_sink_pull_preroll (GstAppSink * appsink)
{
  GstBuffer *buf = NULL;

  //printf("pull_preroll\n");

  g_return_val_if_fail (appsink != NULL, NULL);
  g_return_val_if_fail (GST_IS_APP_SINK (appsink), NULL);

  g_mutex_lock (appsink->mutex);

  while (TRUE) {
    GST_DEBUG_OBJECT (appsink, "trying to grab a buffer");
    if (!appsink->started)
      goto not_started;

    if (appsink->preroll != NULL)
      break;

    if (appsink->is_eos)
      goto eos;

    /* nothing to return, wait */
    GST_DEBUG_OBJECT (appsink, "waiting for the preroll buffer");
    g_cond_wait (appsink->cond, appsink->mutex);
  }
  buf = gst_buffer_ref (appsink->preroll);
  GST_DEBUG_OBJECT (appsink, "we have the preroll buffer %p", buf);
  g_mutex_unlock (appsink->mutex);

  return buf;

  /* special conditions */
eos:
  {
    GST_DEBUG_OBJECT (appsink, "we are EOS, return NULL");
    g_mutex_unlock (appsink->mutex);
    return NULL;
  }
not_started:
  {
    GST_DEBUG_OBJECT (appsink, "we are stopped, return NULL");
    g_mutex_unlock (appsink->mutex);
    return NULL;
  }
}

/**
 * gst_app_sink_pull_buffer:
 * @appsink: a #GstAppSink
 *
 * This function blocks until a buffer or EOS becomes available or the appsink
 * element is set to the READY/NULL state.
 *
 * This function will only return buffers when the appsink is in the PLAYING
 * state. All rendered buffers will be put in a queue so that the application
 * can pull buffers at its own rate. Note that when the application does not
 * pull buffers fast enough, the queued buffers could consume a lot of memory,
 * especially when dealing with raw video frames.
 *
 * If an EOS event was received before any buffers, this function returns
 * %NULL. Use gst_app_sink_is_eos () to check for the EOS condition.
 *
 * Returns: a #GstBuffer or NULL when the appsink is stopped or EOS.
 */
GstBuffer *
gst_app_sink_pull_buffer (GstAppSink * appsink)
{
  GstBuffer *buf = NULL;

  //printf("pull_buffer\n");

  g_return_val_if_fail (appsink != NULL, NULL);
  g_return_val_if_fail (GST_IS_APP_SINK (appsink), NULL);

  g_mutex_lock (appsink->mutex);

  while (TRUE) {
    GST_DEBUG_OBJECT (appsink, "trying to grab a buffer");
    if (!appsink->started)
      goto not_started;

    if (!g_queue_is_empty (appsink->queue))
      break;

    if (appsink->is_eos)
      goto eos;

    /* nothing to return, wait */
    GST_DEBUG_OBJECT (appsink, "waiting for a buffer");
    g_cond_wait (appsink->cond, appsink->mutex);
  }
  buf = (GstBuffer*)g_queue_pop_head (appsink->queue);
  GST_DEBUG_OBJECT (appsink, "we have a buffer %p", buf);
  g_mutex_unlock (appsink->mutex);

  return buf;

  /* special conditions */
eos:
  {
    GST_DEBUG_OBJECT (appsink, "we are EOS, return NULL");
    g_mutex_unlock (appsink->mutex);
    return NULL;
  }
not_started:
  {
    GST_DEBUG_OBJECT (appsink, "we are stopped, return NULL");
    g_mutex_unlock (appsink->mutex);
    return NULL;
  }
}

/**
 * gst_app_sink_peek_buffer:
 * @appsink: a #GstAppSink
 *
 * This function returns a buffer if there is one queued but does not block.
 *
 * This function will only return buffers when the appsink is in the PLAYING
 * state. All rendered buffers will be put in a queue so that the application
 * can pull buffers at its own rate. Note that when the application does not
 * pull buffers fast enough, the queued buffers could consume a lot of memory,
 * especially when dealing with raw video frames.
 *
 * If an EOS event was received before any buffers, this function returns
 * %NULL. Use gst_app_sink_is_eos () to check for the EOS condition.
 *
 * Returns: a #GstBuffer or NULL when the appsink is stopped or EOS.
 */
GstBuffer *
gst_app_sink_peek_buffer (GstAppSink * appsink)
{
  GstBuffer *buf = NULL;

  //printf("pull_buffer\n");

  g_return_val_if_fail (appsink != NULL, NULL);
  g_return_val_if_fail (GST_IS_APP_SINK (appsink), NULL);

  g_mutex_lock (appsink->mutex);

  GST_DEBUG_OBJECT (appsink, "trying to grab a buffer");
  if (!appsink->started)
    goto not_started;

  if (g_queue_is_empty (appsink->queue))
    return NULL;

  if (appsink->is_eos)
    goto eos;

  /* nothing to return, wait */
  buf = (GstBuffer*)g_queue_pop_head (appsink->queue);
  GST_DEBUG_OBJECT (appsink, "we have a buffer %p", buf);
  g_mutex_unlock (appsink->mutex);

  return buf;

  /* special conditions */
eos:
  {
    GST_DEBUG_OBJECT (appsink, "we are EOS, return NULL");
    g_mutex_unlock (appsink->mutex);
    return NULL;
  }
not_started:
  {
    GST_DEBUG_OBJECT (appsink, "we are stopped, return NULL");
    g_mutex_unlock (appsink->mutex);
    return NULL;
  }
}

guint
gst_app_sink_get_queue_length (GstAppSink * appsink)
{
  return g_queue_get_length (appsink->queue);
}
#endif
