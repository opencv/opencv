/****************************************************************************
** Meta object code from reading C++ file 'window_QT.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../modules/highgui/src/window_QT.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'window_QT.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_GuiReceiver[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
      28,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      24,   13,   12,   12, 0x0a,
      55,   50,   12,   12, 0x2a,
      77,   50,   12,   12, 0x0a,
     100,   12,   12,   12, 0x0a,
     167,  119,   12,   12, 0x0a,
     267,  210,   12,   12, 0x0a,
     326,  317,   12,   12, 0x0a,
     372,  354,   12,   12, 0x0a,
     411,  402,   12,   12, 0x0a,
     454,  436,   12,   12, 0x0a,
     487,  436,   12,   12, 0x0a,
     525,   12,   12,   12, 0x0a,
     535,   13,   12,   12, 0x0a,
     575,   50,  568,   12, 0x0a,
     597,   50,  568,   12, 0x0a,
     620,   13,   12,   12, 0x0a,
     650,   50,  568,   12, 0x0a,
     684,  674,   12,   12, 0x0a,
     715,   50,   12,   12, 0x0a,
     745,   50,   12,   12, 0x0a,
     794,  775,   12,   12, 0x0a,
     894,  830,   12,   12, 0x0a,
     933,   12,   12,   12, 0x0a,
     991,  968,   12,   12, 0x0a,
    1034,  968,   12,   12, 0x0a,
    1078,   50,   12,   12, 0x0a,
    1104,   50,   12,   12, 0x0a,
    1126,   50,  568,   12, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_GuiReceiver[] = {
    "GuiReceiver\0\0name,flags\0"
    "createWindow(QString,int)\0name\0"
    "createWindow(QString)\0destroyWindow(QString)\0"
    "destroyAllWindow()\0"
    "trackbar_name,window_name,value,count,on_change\0"
    "addSlider(QString,QString,void*,int,void*)\0"
    "trackbar_name,window_name,value,count,on_change,userdata\0"
    "addSlider2(QString,QString,void*,int,void*,void*)\0"
    "name,x,y\0moveWindow(QString,int,int)\0"
    "name,width,height\0resizeWindow(QString,int,int)\0"
    "name,arr\0showImage(QString,void*)\0"
    "name,text,delayms\0displayInfo(QString,QString,int)\0"
    "displayStatusBar(QString,QString,int)\0"
    "timeOut()\0toggleFullScreen(QString,double)\0"
    "double\0isFullScreen(QString)\0"
    "getPropWindow(QString)\0"
    "setPropWindow(QString,double)\0"
    "getRatioWindow(QString)\0name,arg2\0"
    "setRatioWindow(QString,double)\0"
    "saveWindowParameters(QString)\0"
    "loadWindowParameters(QString)\0"
    "arg1,text,org,font\0"
    "putText(void*,QString,QPoint,void*)\0"
    "button_name,button_type,initial_button_state,on_change,userdata\0"
    "addButton(QString,int,int,void*,void*)\0"
    "enablePropertiesButtonEachWindow()\0"
    "name,callback,userdata\0"
    "setOpenGlDrawCallback(QString,void*,void*)\0"
    "setOpenGlCleanCallback(QString,void*,void*)\0"
    "setOpenGlContext(QString)\0"
    "updateWindow(QString)\0isOpenGl(QString)\0"
};

void GuiReceiver::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        GuiReceiver *_t = static_cast<GuiReceiver *>(_o);
        switch (_id) {
        case 0: _t->createWindow((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 1: _t->createWindow((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 2: _t->destroyWindow((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 3: _t->destroyAllWindow(); break;
        case 4: _t->addSlider((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2])),(*reinterpret_cast< void*(*)>(_a[3])),(*reinterpret_cast< int(*)>(_a[4])),(*reinterpret_cast< void*(*)>(_a[5]))); break;
        case 5: _t->addSlider2((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2])),(*reinterpret_cast< void*(*)>(_a[3])),(*reinterpret_cast< int(*)>(_a[4])),(*reinterpret_cast< void*(*)>(_a[5])),(*reinterpret_cast< void*(*)>(_a[6]))); break;
        case 6: _t->moveWindow((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3]))); break;
        case 7: _t->resizeWindow((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3]))); break;
        case 8: _t->showImage((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< void*(*)>(_a[2]))); break;
        case 9: _t->displayInfo((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3]))); break;
        case 10: _t->displayStatusBar((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3]))); break;
        case 11: _t->timeOut(); break;
        case 12: _t->toggleFullScreen((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2]))); break;
        case 13: { double _r = _t->isFullScreen((*reinterpret_cast< QString(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< double*>(_a[0]) = _r; }  break;
        case 14: { double _r = _t->getPropWindow((*reinterpret_cast< QString(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< double*>(_a[0]) = _r; }  break;
        case 15: _t->setPropWindow((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2]))); break;
        case 16: { double _r = _t->getRatioWindow((*reinterpret_cast< QString(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< double*>(_a[0]) = _r; }  break;
        case 17: _t->setRatioWindow((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2]))); break;
        case 18: _t->saveWindowParameters((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 19: _t->loadWindowParameters((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 20: _t->putText((*reinterpret_cast< void*(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2])),(*reinterpret_cast< QPoint(*)>(_a[3])),(*reinterpret_cast< void*(*)>(_a[4]))); break;
        case 21: _t->addButton((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< int(*)>(_a[3])),(*reinterpret_cast< void*(*)>(_a[4])),(*reinterpret_cast< void*(*)>(_a[5]))); break;
        case 22: _t->enablePropertiesButtonEachWindow(); break;
        case 23: _t->setOpenGlDrawCallback((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< void*(*)>(_a[2])),(*reinterpret_cast< void*(*)>(_a[3]))); break;
        case 24: _t->setOpenGlCleanCallback((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< void*(*)>(_a[2])),(*reinterpret_cast< void*(*)>(_a[3]))); break;
        case 25: _t->setOpenGlContext((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 26: _t->updateWindow((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 27: { double _r = _t->isOpenGl((*reinterpret_cast< QString(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< double*>(_a[0]) = _r; }  break;
        default: ;
        }
    }
}

const QMetaObjectExtraData GuiReceiver::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject GuiReceiver::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_GuiReceiver,
      qt_meta_data_GuiReceiver, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &GuiReceiver::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *GuiReceiver::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *GuiReceiver::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_GuiReceiver))
        return static_cast<void*>(const_cast< GuiReceiver*>(this));
    return QObject::qt_metacast(_clname);
}

int GuiReceiver::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 28)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 28;
    }
    return _id;
}
static const uint qt_meta_data_CvButtonbar[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

static const char qt_meta_stringdata_CvButtonbar[] = {
    "CvButtonbar\0"
};

void CvButtonbar::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    Q_UNUSED(_o);
    Q_UNUSED(_id);
    Q_UNUSED(_c);
    Q_UNUSED(_a);
}

const QMetaObjectExtraData CvButtonbar::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject CvButtonbar::staticMetaObject = {
    { &CvBar::staticMetaObject, qt_meta_stringdata_CvButtonbar,
      qt_meta_data_CvButtonbar, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &CvButtonbar::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *CvButtonbar::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *CvButtonbar::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_CvButtonbar))
        return static_cast<void*>(const_cast< CvButtonbar*>(this));
    return CvBar::qt_metacast(_clname);
}

int CvButtonbar::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = CvBar::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    return _id;
}
static const uint qt_meta_data_CvPushButton[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      14,   13,   13,   13, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_CvPushButton[] = {
    "CvPushButton\0\0callCallBack(bool)\0"
};

void CvPushButton::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        CvPushButton *_t = static_cast<CvPushButton *>(_o);
        switch (_id) {
        case 0: _t->callCallBack((*reinterpret_cast< bool(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData CvPushButton::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject CvPushButton::staticMetaObject = {
    { &QPushButton::staticMetaObject, qt_meta_stringdata_CvPushButton,
      qt_meta_data_CvPushButton, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &CvPushButton::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *CvPushButton::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *CvPushButton::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_CvPushButton))
        return static_cast<void*>(const_cast< CvPushButton*>(this));
    return QPushButton::qt_metacast(_clname);
}

int CvPushButton::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QPushButton::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    }
    return _id;
}
static const uint qt_meta_data_CvCheckBox[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      12,   11,   11,   11, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_CvCheckBox[] = {
    "CvCheckBox\0\0callCallBack(bool)\0"
};

void CvCheckBox::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        CvCheckBox *_t = static_cast<CvCheckBox *>(_o);
        switch (_id) {
        case 0: _t->callCallBack((*reinterpret_cast< bool(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData CvCheckBox::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject CvCheckBox::staticMetaObject = {
    { &QCheckBox::staticMetaObject, qt_meta_stringdata_CvCheckBox,
      qt_meta_data_CvCheckBox, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &CvCheckBox::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *CvCheckBox::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *CvCheckBox::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_CvCheckBox))
        return static_cast<void*>(const_cast< CvCheckBox*>(this));
    return QCheckBox::qt_metacast(_clname);
}

int CvCheckBox::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QCheckBox::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    }
    return _id;
}
static const uint qt_meta_data_CvRadioButton[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      15,   14,   14,   14, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_CvRadioButton[] = {
    "CvRadioButton\0\0callCallBack(bool)\0"
};

void CvRadioButton::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        CvRadioButton *_t = static_cast<CvRadioButton *>(_o);
        switch (_id) {
        case 0: _t->callCallBack((*reinterpret_cast< bool(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData CvRadioButton::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject CvRadioButton::staticMetaObject = {
    { &QRadioButton::staticMetaObject, qt_meta_stringdata_CvRadioButton,
      qt_meta_data_CvRadioButton, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &CvRadioButton::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *CvRadioButton::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *CvRadioButton::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_CvRadioButton))
        return static_cast<void*>(const_cast< CvRadioButton*>(this));
    return QRadioButton::qt_metacast(_clname);
}

int CvRadioButton::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QRadioButton::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    }
    return _id;
}
static const uint qt_meta_data_CvTrackbar[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       2,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      12,   11,   11,   11, 0x08,
      35,   27,   11,   11, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_CvTrackbar[] = {
    "CvTrackbar\0\0createDialog()\0myvalue\0"
    "update(int)\0"
};

void CvTrackbar::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        CvTrackbar *_t = static_cast<CvTrackbar *>(_o);
        switch (_id) {
        case 0: _t->createDialog(); break;
        case 1: _t->update((*reinterpret_cast< int(*)>(_a[1]))); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData CvTrackbar::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject CvTrackbar::staticMetaObject = {
    { &CvBar::staticMetaObject, qt_meta_stringdata_CvTrackbar,
      qt_meta_data_CvTrackbar, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &CvTrackbar::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *CvTrackbar::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *CvTrackbar::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_CvTrackbar))
        return static_cast<void*>(const_cast< CvTrackbar*>(this));
    return CvBar::qt_metacast(_clname);
}

int CvTrackbar::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = CvBar::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 2)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 2;
    }
    return _id;
}
static const uint qt_meta_data_CvWinProperties[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

       0        // eod
};

static const char qt_meta_stringdata_CvWinProperties[] = {
    "CvWinProperties\0"
};

void CvWinProperties::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    Q_UNUSED(_o);
    Q_UNUSED(_id);
    Q_UNUSED(_c);
    Q_UNUSED(_a);
}

const QMetaObjectExtraData CvWinProperties::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject CvWinProperties::staticMetaObject = {
    { &CvWinModel::staticMetaObject, qt_meta_stringdata_CvWinProperties,
      qt_meta_data_CvWinProperties, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &CvWinProperties::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *CvWinProperties::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *CvWinProperties::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_CvWinProperties))
        return static_cast<void*>(const_cast< CvWinProperties*>(this));
    return CvWinModel::qt_metacast(_clname);
}

int CvWinProperties::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = CvWinModel::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    return _id;
}
static const uint qt_meta_data_CvWindow[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      10,    9,    9,    9, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_CvWindow[] = {
    "CvWindow\0\0displayPropertiesWin()\0"
};

void CvWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        CvWindow *_t = static_cast<CvWindow *>(_o);
        switch (_id) {
        case 0: _t->displayPropertiesWin(); break;
        default: ;
        }
    }
    Q_UNUSED(_a);
}

const QMetaObjectExtraData CvWindow::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject CvWindow::staticMetaObject = {
    { &CvWinModel::staticMetaObject, qt_meta_stringdata_CvWindow,
      qt_meta_data_CvWindow, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &CvWindow::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *CvWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *CvWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_CvWindow))
        return static_cast<void*>(const_cast< CvWindow*>(this));
    return CvWinModel::qt_metacast(_clname);
}

int CvWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = CvWinModel::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 1)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 1;
    }
    return _id;
}
static const uint qt_meta_data_DefaultViewPort[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
      10,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      17,   16,   16,   16, 0x0a,
      36,   16,   16,   16, 0x0a,
      56,   16,   16,   16, 0x0a,
      73,   16,   16,   16, 0x0a,
      92,   16,   16,   16, 0x0a,
     104,   16,   16,   16, 0x0a,
     116,   16,   16,   16, 0x0a,
     125,   16,   16,   16, 0x0a,
     135,   16,   16,   16, 0x0a,
     146,   16,   16,   16, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_DefaultViewPort[] = {
    "DefaultViewPort\0\0siftWindowOnLeft()\0"
    "siftWindowOnRight()\0siftWindowOnUp()\0"
    "siftWindowOnDown()\0resetZoom()\0"
    "imgRegion()\0ZoomIn()\0ZoomOut()\0"
    "saveView()\0stopDisplayInfo()\0"
};

void DefaultViewPort::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        DefaultViewPort *_t = static_cast<DefaultViewPort *>(_o);
        switch (_id) {
        case 0: _t->siftWindowOnLeft(); break;
        case 1: _t->siftWindowOnRight(); break;
        case 2: _t->siftWindowOnUp(); break;
        case 3: _t->siftWindowOnDown(); break;
        case 4: _t->resetZoom(); break;
        case 5: _t->imgRegion(); break;
        case 6: _t->ZoomIn(); break;
        case 7: _t->ZoomOut(); break;
        case 8: _t->saveView(); break;
        case 9: _t->stopDisplayInfo(); break;
        default: ;
        }
    }
    Q_UNUSED(_a);
}

const QMetaObjectExtraData DefaultViewPort::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject DefaultViewPort::staticMetaObject = {
    { &QGraphicsView::staticMetaObject, qt_meta_stringdata_DefaultViewPort,
      qt_meta_data_DefaultViewPort, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &DefaultViewPort::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *DefaultViewPort::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *DefaultViewPort::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_DefaultViewPort))
        return static_cast<void*>(const_cast< DefaultViewPort*>(this));
    if (!strcmp(_clname, "ViewPort"))
        return static_cast< ViewPort*>(const_cast< DefaultViewPort*>(this));
    return QGraphicsView::qt_metacast(_clname);
}

int DefaultViewPort::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QGraphicsView::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 10)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 10;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
