#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>

#include "common.hpp"

float vertexes[] = {
        -1.0f, -1.0f,
        -1.0f,  1.0f,
         1.0f, -1.0f,
         1.0f,  1.0f
};
float texCoordOES[] = {
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 1.0f,
        1.0f, 0.0f
};
float texCoord2D[] = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
};

const char vss[] = \
            "attribute vec2 vPosition;\n" \
            "attribute vec2 vTexCoord;\n" \
            "varying vec2 texCoord;\n" \
            "void main() {\n" \
            "  texCoord = vTexCoord;\n" \
            "  gl_Position = vec4 ( vPosition.x, vPosition.y, 0.0, 1.0 );\n" \
            "}";

const char fssOES[] = \
            "#extension GL_OES_EGL_image_external : require\n" \
            "precision mediump float;\n" \
            "uniform samplerExternalOES sTexture;\n" \
            "varying vec2 texCoord;\n" \
            "void main() {\n" \
            "  gl_FragColor = texture2D(sTexture,texCoord);\n" \
            "}";

const char fss2D[] = \
            "precision mediump float;\n" \
            "uniform sampler2D sTexture;\n" \
            "varying vec2 texCoord;\n" \
            "void main() {\n" \
            "  gl_FragColor = texture2D(sTexture,texCoord);\n" \
            "}";

int progOES = 0;
int prog2D = 0;

GLuint FBOtex = 0, FBOtex2 = 0;
GLuint FBO = 0;

GLuint texOES = 0;
int texWidth = 0, texHeight = 0;

static inline void deleteTex(GLuint* tex)
{
    if(tex && *tex)
    {
        glDeleteTextures(1, tex);
        *tex = 0;
    }
}

static void releaseFBO()
{
    if (FBO != 0)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteFramebuffers(1, &FBO);
        FBO = 0;
    }
    deleteTex(&FBOtex);
    deleteTex(&FBOtex2);
    glDeleteProgram(prog2D);
    prog2D = 0;
}

static inline void logShaderCompileError(GLuint shader)
{
    GLchar msg[512];
    msg[0] = 0;
    GLsizei len;
    glGetShaderInfoLog(shader, sizeof(msg) - 1, &len, msg);
    LOGE("Could not compile shader: %s", msg);
}

static int makeShaderProg(const char* vss, const char* fss)
{
    LOGD("makeShaderProg: setup GL_VERTEX_SHADER");
    GLuint vshader = glCreateShader(GL_VERTEX_SHADER);
    const GLchar* text = vss;
    glShaderSource(vshader, 1, &text, 0);
    glCompileShader(vshader);
    int compiled;
    glGetShaderiv(vshader, GL_COMPILE_STATUS, &compiled);
    if (compiled == 0) {
        logShaderCompileError(vshader);
        glDeleteShader(vshader);
        vshader = 0;
    }

    LOGD("makeShaderProg: setup GL_FRAGMENT_SHADER");
    GLuint fshader = glCreateShader(GL_FRAGMENT_SHADER);
    text = fss;
    glShaderSource(fshader, 1, &text, 0);
    glCompileShader(fshader);
    glGetShaderiv(fshader, GL_COMPILE_STATUS, &compiled);
    if (compiled == 0) {
        logShaderCompileError(fshader);
        glDeleteShader(fshader);
        fshader = 0;
    }

    LOGD("makeShaderProg: glCreateProgram");
    GLuint program = glCreateProgram();
    glAttachShader(program, vshader);
    glAttachShader(program, fshader);
    glLinkProgram(program);

    if(vshader) glDeleteShader(vshader);
    if(fshader) glDeleteShader(fshader);

    return program;
}


static void initFBO(int width, int height)
{
    LOGD("initFBO(%d, %d)", width, height);
    releaseFBO();

    glGenTextures(1, &FBOtex2);
    glBindTexture(GL_TEXTURE_2D, FBOtex2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenTextures(1, &FBOtex);
    glBindTexture(GL_TEXTURE_2D, FBOtex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    //int hFBO;
    glGenFramebuffers(1, &FBO);
    glBindFramebuffer(GL_FRAMEBUFFER, FBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, FBOtex, 0);
    LOGD("initFBO status: %d", glGetError());

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        LOGE("initFBO failed: %d", glCheckFramebufferStatus(GL_FRAMEBUFFER));

    prog2D = makeShaderProg(vss, fss2D);
}

void drawTex(int tex, GLenum texType, GLuint fbo)
{
    int64_t t = getTimeMs();
    //draw texture to FBO or to screen
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, texWidth, texHeight);

    glClear(GL_COLOR_BUFFER_BIT);

    int prog = texType == GL_TEXTURE_EXTERNAL_OES ? progOES : prog2D;
    glUseProgram(prog);
    int vPos  = glGetAttribLocation(prog, "vPosition");
    int vTC = glGetAttribLocation(prog, "vTexCoord");

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(texType, tex);
    glUniform1i(glGetUniformLocation(prog, "sTexture"), 0);

    glVertexAttribPointer(vPos, 2, GL_FLOAT, false, 4*2, vertexes);
    glVertexAttribPointer(vTC,  2, GL_FLOAT, false, 4*2, texType == GL_TEXTURE_EXTERNAL_OES ? texCoordOES : texCoord2D);
    glEnableVertexAttribArray(vPos);
    glEnableVertexAttribArray(vTC);

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glFlush();
    LOGD("drawTex(%u) costs %d ms", tex, getTimeInterval(t));
}

void drawFrameOrig()
{
    drawTex(texOES, GL_TEXTURE_EXTERNAL_OES, 0);
}

void procCPU(char* buff, int w, int h)
{
    int64_t t = getTimeMs();
    for(int i=0; i<h; i++)
    {
        buff[i*w*4+i*4+0] = 255;
        buff[i*w*4+i*4+4] = 255;
        buff[i*w*4+i*4+8] = 255;
    }
    LOGD("procCPU() costs %d ms", getTimeInterval(t));
}

void drawFrameProcCPU()
{
    int64_t t;
    drawTex(texOES, GL_TEXTURE_EXTERNAL_OES, FBO);

    // let's modify pixels in FBO texture in C++ code (on CPU)
    const int BUFF_SIZE = 1<<24;//2k*2k*4;
    static char tmpBuff[BUFF_SIZE];
    if(texWidth*texHeight > BUFF_SIZE)
    {
        LOGE("Internal temp buffer is too small, can't make CPU frame processing");
        return;
    }

    // read
    t = getTimeMs();
    glReadPixels(0, 0, texWidth, texHeight, GL_RGBA, GL_UNSIGNED_BYTE, tmpBuff);
    LOGD("glReadPixels() costs %d ms", getTimeInterval(t));

   // modify
    procCPU(tmpBuff, texWidth, texHeight);

    // write back
    t = getTimeMs();
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, texWidth, texHeight, GL_RGBA, GL_UNSIGNED_BYTE, tmpBuff);
    LOGD("glTexSubImage2D() costs %d ms", getTimeInterval(t));

    // render to screen
    drawTex(FBOtex, GL_TEXTURE_2D, 0);
}

void procOCL(int tex, int w, int h);
void procOCL_I2I(int texIn, int texOut, int w, int h);
void drawFrameProcOCL()
{
    drawTex(texOES, GL_TEXTURE_EXTERNAL_OES, FBO);

    // modify pixels in FBO texture using OpenCL and CL-GL interop
    procOCL_I2I(FBOtex, FBOtex2, texWidth, texHeight);

    // render to screen
    drawTex(FBOtex2, GL_TEXTURE_2D, 0);
}


extern "C" void drawFrame()
{
    LOGD("*** drawFrame() ***");
    int64_t t = getTimeMs();
    //drawFrameOrig();
    //drawFrameProcCPU();
    drawFrameProcOCL();
    glFinish();
    LOGD("*** drawFrame() costs %d ms ***", getTimeInterval(t));
}

void closeCL();
extern "C" void closeGL()
{
    closeCL();
    LOGD("closeGL");
    deleteTex(&texOES);

    glUseProgram(0);
    glDeleteProgram(progOES);
    progOES = 0;

    releaseFBO();
}

void initCL();
extern "C" int initGL()
{
    LOGD("initGL");

    closeGL();

    const char* vs = (const char*)glGetString(GL_VERSION);
    LOGD("GL_VERSION = %s", vs);

    progOES = makeShaderProg(vss, fssOES);

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    texOES = 0;
    glGenTextures(1, &texOES);
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, texOES);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    initCL();

    return texOES;
}

extern "C" void changeSize(int width, int height)
{
    const int MAX_W=1<<11, MAX_H=1<<11;
    LOGD("changeSize: %dx%d", width, height);
    texWidth = width <= MAX_W ? width : MAX_W;
    texHeight = height <= MAX_H ? height : MAX_H;
    initFBO(texWidth, texHeight);
}
