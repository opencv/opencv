#!/usr/bin/env python
""" Utility package used by other test result formatting scripts.
"""
import math, os, sys

webcolors = {
"indianred": "#cd5c5c",
"lightcoral": "#f08080",
"salmon": "#fa8072",
"darksalmon": "#e9967a",
"lightsalmon": "#ffa07a",
"red": "#ff0000",
"crimson": "#dc143c",
"firebrick": "#b22222",
"darkred": "#8b0000",
"pink": "#ffc0cb",
"lightpink": "#ffb6c1",
"hotpink": "#ff69b4",
"deeppink": "#ff1493",
"mediumvioletred": "#c71585",
"palevioletred": "#db7093",
"lightsalmon": "#ffa07a",
"coral": "#ff7f50",
"tomato": "#ff6347",
"orangered": "#ff4500",
"darkorange": "#ff8c00",
"orange": "#ffa500",
"gold": "#ffd700",
"yellow": "#ffff00",
"lightyellow": "#ffffe0",
"lemonchiffon": "#fffacd",
"lightgoldenrodyellow": "#fafad2",
"papayawhip": "#ffefd5",
"moccasin": "#ffe4b5",
"peachpuff": "#ffdab9",
"palegoldenrod": "#eee8aa",
"khaki": "#f0e68c",
"darkkhaki": "#bdb76b",
"lavender": "#e6e6fa",
"thistle": "#d8bfd8",
"plum": "#dda0dd",
"violet": "#ee82ee",
"orchid": "#da70d6",
"fuchsia": "#ff00ff",
"magenta": "#ff00ff",
"mediumorchid": "#ba55d3",
"mediumpurple": "#9370db",
"blueviolet": "#8a2be2",
"darkviolet": "#9400d3",
"darkorchid": "#9932cc",
"darkmagenta": "#8b008b",
"purple": "#800080",
"indigo": "#4b0082",
"darkslateblue": "#483d8b",
"slateblue": "#6a5acd",
"mediumslateblue": "#7b68ee",
"greenyellow": "#adff2f",
"chartreuse": "#7fff00",
"lawngreen": "#7cfc00",
"lime": "#00ff00",
"limegreen": "#32cd32",
"palegreen": "#98fb98",
"lightgreen": "#90ee90",
"mediumspringgreen": "#00fa9a",
"springgreen": "#00ff7f",
"mediumseagreen": "#3cb371",
"seagreen": "#2e8b57",
"forestgreen": "#228b22",
"green": "#008000",
"darkgreen": "#006400",
"yellowgreen": "#9acd32",
"olivedrab": "#6b8e23",
"olive": "#808000",
"darkolivegreen": "#556b2f",
"mediumaquamarine": "#66cdaa",
"darkseagreen": "#8fbc8f",
"lightseagreen": "#20b2aa",
"darkcyan": "#008b8b",
"teal": "#008080",
"aqua": "#00ffff",
"cyan": "#00ffff",
"lightcyan": "#e0ffff",
"paleturquoise": "#afeeee",
"aquamarine": "#7fffd4",
"turquoise": "#40e0d0",
"mediumturquoise": "#48d1cc",
"darkturquoise": "#00ced1",
"cadetblue": "#5f9ea0",
"steelblue": "#4682b4",
"lightsteelblue": "#b0c4de",
"powderblue": "#b0e0e6",
"lightblue": "#add8e6",
"skyblue": "#87ceeb",
"lightskyblue": "#87cefa",
"deepskyblue": "#00bfff",
"dodgerblue": "#1e90ff",
"cornflowerblue": "#6495ed",
"royalblue": "#4169e1",
"blue": "#0000ff",
"mediumblue": "#0000cd",
"darkblue": "#00008b",
"navy": "#000080",
"midnightblue": "#191970",
"cornsilk": "#fff8dc",
"blanchedalmond": "#ffebcd",
"bisque": "#ffe4c4",
"navajowhite": "#ffdead",
"wheat": "#f5deb3",
"burlywood": "#deb887",
"tan": "#d2b48c",
"rosybrown": "#bc8f8f",
"sandybrown": "#f4a460",
"goldenrod": "#daa520",
"darkgoldenrod": "#b8860b",
"peru": "#cd853f",
"chocolate": "#d2691e",
"saddlebrown": "#8b4513",
"sienna": "#a0522d",
"brown": "#a52a2a",
"maroon": "#800000",
"white": "#ffffff",
"snow": "#fffafa",
"honeydew": "#f0fff0",
"mintcream": "#f5fffa",
"azure": "#f0ffff",
"aliceblue": "#f0f8ff",
"ghostwhite": "#f8f8ff",
"whitesmoke": "#f5f5f5",
"seashell": "#fff5ee",
"beige": "#f5f5dc",
"oldlace": "#fdf5e6",
"floralwhite": "#fffaf0",
"ivory": "#fffff0",
"antiquewhite": "#faebd7",
"linen": "#faf0e6",
"lavenderblush": "#fff0f5",
"mistyrose": "#ffe4e1",
"gainsboro": "#dcdcdc",
"lightgrey": "#d3d3d3",
"silver": "#c0c0c0",
"darkgray": "#a9a9a9",
"gray": "#808080",
"dimgray": "#696969",
"lightslategray": "#778899",
"slategray": "#708090",
"darkslategray": "#2f4f4f",
"black": "#000000",
}

if os.name == "nt":
    consoleColors = [
    "#000000",  #{   0,   0,   0 },//0 - black
    "#000080",  #{   0,   0, 128 },//1 - navy
    "#008000",  #{   0, 128,   0 },//2 - green
    "#008080",  #{   0, 128, 128 },//3 - teal
    "#800000",  #{ 128,   0,   0 },//4 - maroon
    "#800080",  #{ 128,   0, 128 },//5 - purple
    "#808000",  #{ 128, 128,   0 },//6 - olive
    "#C0C0C0",  #{ 192, 192, 192 },//7 - silver
    "#808080",  #{ 128, 128, 128 },//8 - gray
    "#0000FF",  #{   0,   0, 255 },//9 - blue
    "#00FF00",  #{   0, 255,   0 },//a - lime
    "#00FFFF",  #{   0, 255, 255 },//b - cyan
    "#FF0000",  #{ 255,   0,   0 },//c - red
    "#FF00FF",  #{ 255,   0, 255 },//d - magenta
    "#FFFF00",  #{ 255, 255,   0 },//e - yellow
    "#FFFFFF",  #{ 255, 255, 255 } //f - white
    ]
else:
    consoleColors = [
    "#2e3436",
    "#cc0000",
    "#4e9a06",
    "#c4a000",
    "#3465a4",
    "#75507b",
    "#06989a",
    "#d3d7cf",
    "#ffffff",

    "#555753",
    "#ef2929",
    "#8ae234",
    "#fce94f",
    "#729fcf",
    "#ad7fa8",
    "#34e2e2",
    "#eeeeec",
    ]

def RGB2LAB(r,g,b):
    if max(r,g,b):
        r /= 255.
        g /= 255.
        b /= 255.

    X = (0.412453 * r + 0.357580 * g + 0.180423 * b) / 0.950456
    Y = (0.212671 * r + 0.715160 * g + 0.072169 * b)
    Z = (0.019334 * r + 0.119193 * g + 0.950227 * b) / 1.088754

    #[X * 0.950456]   [0.412453 0.357580 0.180423]   [R]
    #[Y           ] = [0.212671 0.715160 0.072169] * [G]
    #[Z * 1.088754]   [0.019334 0.119193 0.950227]   [B]

    T = 0.008856 #threshold

    if X > T:
        fX = math.pow(X, 1./3.)
    else:
        fX = 7.787 * X + 16./116.

    # Compute L
    if Y > T:
        Y3 = math.pow(Y, 1./3.)
        fY = Y3
        L  = 116. * Y3 - 16.0
    else:
        fY = 7.787 * Y + 16./116.
        L  = 903.3 * Y

    if Z > T:
        fZ = math.pow(Z, 1./3.)
    else:
        fZ = 7.787 * Z + 16./116.

    # Compute a and b
    a = 500. * (fX - fY)
    b = 200. * (fY - fZ)

    return (L,a,b)

def colorDistance(r1,g1,b1 = None, r2 = None, g2 = None,b2 = None):
    if type(r1) == tuple and type(g1) == tuple and b1 is None and r2 is None and g2 is None and b2 is None:
        (l1,a1,b1) = RGB2LAB(*r1)
        (l2,a2,b2) = RGB2LAB(*g1)
    else:
        (l1,a1,b1) = RGB2LAB(r1,g1,b1)
        (l2,a2,b2) = RGB2LAB(r2,g2,b2)
    #CIE94
    dl = l1-l2
    C1 = math.sqrt(a1*a1 + b1*b1)
    C2 = math.sqrt(a2*a2 + b2*b2)
    dC = C1 - C2
    da = a1-a2
    db = b1-b2
    dH = math.sqrt(max(0, da*da + db*db - dC*dC))
    Kl = 1
    K1 = 0.045
    K2 = 0.015

    s1 = dl/Kl
    s2 = dC/(1. + K1 * C1)
    s3 = dH/(1. + K2 * C1)
    return math.sqrt(s1*s1 + s2*s2 + s3*s3)

def parseHexColor(col):
    if len(col) != 4 and len(col) != 7 and not col.startswith("#"):
        return (0,0,0)
    if len(col) == 4:
        r = col[1]*2
        g = col[2]*2
        b = col[3]*2
    else:
        r = col[1:3]
        g = col[3:5]
        b = col[5:7]
    return (int(r,16), int(g,16), int(b,16))

def getColor(col):
    if isinstance(col, str):
        if col.lower() in webcolors:
            return parseHexColor(webcolors[col.lower()])
        else:
            return parseHexColor(col)
    else:
        return col

def getNearestConsoleColor(col):
    color = getColor(col)
    minidx = 0
    mindist = colorDistance(color, getColor(consoleColors[0]))
    for i in range(len(consoleColors)):
        dist = colorDistance(color, getColor(consoleColors[i]))
        if dist < mindist:
            mindist = dist
            minidx = i
    return minidx

if os.name == 'nt':
    import msvcrt
    from ctypes import windll, Structure, c_short, c_ushort, byref
    SHORT = c_short
    WORD = c_ushort

    class COORD(Structure):
        _fields_ = [
            ("X", SHORT),
            ("Y", SHORT)]

    class SMALL_RECT(Structure):
        _fields_ = [
            ("Left", SHORT),
            ("Top", SHORT),
            ("Right", SHORT),
            ("Bottom", SHORT)]

    class CONSOLE_SCREEN_BUFFER_INFO(Structure):
        _fields_ = [
            ("dwSize", COORD),
            ("dwCursorPosition", COORD),
            ("wAttributes", WORD),
            ("srWindow", SMALL_RECT),
            ("dwMaximumWindowSize", COORD)]

    class winConsoleColorizer(object):
        def __init__(self, stream):
            self.handle = msvcrt.get_osfhandle(stream.fileno())
            self.default_attrs = 7#self.get_text_attr()
            self.stream = stream

        def get_text_attr(self):
            csbi = CONSOLE_SCREEN_BUFFER_INFO()
            windll.kernel32.GetConsoleScreenBufferInfo(self.handle, byref(csbi))
            return csbi.wAttributes

        def set_text_attr(self, color):
            windll.kernel32.SetConsoleTextAttribute(self.handle, color)

        def write(self, *text, **attrs):
            if not text:
                return
            color = attrs.get("color", None)
            if color:
                col = getNearestConsoleColor(color)
                self.stream.flush()
                self.set_text_attr(col)
            self.stream.write(" ".join([str(t) for t in text]))
            if color:
                self.stream.flush()
                self.set_text_attr(self.default_attrs)

class dummyColorizer(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, *text, **attrs):
        if text:
            self.stream.write(" ".join([str(t) for t in text]))

class asciiSeqColorizer(object):
    RESET_SEQ = "\033[0m"
    #BOLD_SEQ = "\033[1m"
    ITALIC_SEQ = "\033[3m"
    UNDERLINE_SEQ = "\033[4m"
    STRIKEOUT_SEQ = "\033[9m"
    COLOR_SEQ0 = "\033[00;%dm" #dark
    COLOR_SEQ1 = "\033[01;%dm" #bold and light

    def __init__(self, stream):
        self.stream = stream

    def get_seq(self, code):
        if code > 8:
            return self.__class__.COLOR_SEQ1 % (30 + code - 9)
        else:
            return self.__class__.COLOR_SEQ0 % (30 + code)

    def write(self, *text, **attrs):
        if not text:
            return
        color = attrs.get("color", None)
        if color:
            col = getNearestConsoleColor(color)
            self.stream.write(self.get_seq(col))
        self.stream.write(" ".join([str(t) for t in text]))
        if color:
            self.stream.write(self.__class__.RESET_SEQ)


def getColorizer(stream):
    if stream.isatty():
        if os.name == "nt":
            return winConsoleColorizer(stream)
        else:
            return asciiSeqColorizer(stream)
    else:
        return dummyColorizer(stream)
