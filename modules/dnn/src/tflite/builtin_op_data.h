// source: https://github.com/tensorflow/tensorflow/blob/b2f5959ff823a8ed5bf4883e785f8f96d4253a8b/tensorflow/lite/core/c/builtin_op_data.h
typedef enum {
    kTfLitePaddingUnknown = 0,
    kTfLitePaddingSame,
    kTfLitePaddingValid,
} TfLitePadding;

typedef enum {
    kTfLiteActNone = 0,
    kTfLiteActRelu,
    kTfLiteActReluN1To1,  // min(max(-1, x), 1)
    kTfLiteActRelu6,      // min(max(0, x), 6)
    kTfLiteActTanh,
    kTfLiteActSignBit,
    kTfLiteActSigmoid,
} TfLiteFusedActivation;

typedef struct {
    int width;
    int height;
    int width_offset;
    int height_offset;
} TfLitePaddingValues;

typedef struct {
    TfLitePadding padding;
    int stride_width;
    int stride_height;
    int filter_width;
    int filter_height;
    TfLiteFusedActivation activation;
    struct {
        TfLitePaddingValues padding;
    } computed;
} TfLitePoolParams;

typedef struct {
    TfLitePadding padding;
    int stride_width;
    int stride_height;
} TfLiteTransposeConvParams;
