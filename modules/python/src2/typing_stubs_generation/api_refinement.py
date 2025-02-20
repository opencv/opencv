__all__ = [
    "apply_manual_api_refinement"
]

from typing import cast, Sequence, Callable, Iterable

from .nodes import (NamespaceNode, FunctionNode, OptionalTypeNode, TypeNode,
                    ClassProperty, PrimitiveTypeNode, ASTNodeTypeNode,
                    AggregatedTypeNode, CallableTypeNode, AnyTypeNode,
                    TupleTypeNode, UnionTypeNode, ProtocolClassNode,
                    DictTypeNode, ClassTypeNode)
from .ast_utils import (find_function_node, SymbolName,
                        for_each_function_overload)
from .types_conversion import create_type_node


def apply_manual_api_refinement(root: NamespaceNode) -> None:
    refine_highgui_module(root)
    refine_cuda_module(root)
    export_matrix_type_constants(root)
    refine_dnn_module(root)
    # Export OpenCV exception class
    builtin_exception = root.add_class("Exception")
    builtin_exception.is_exported = False
    root.add_class("error", (builtin_exception, ), ERROR_CLASS_PROPERTIES)
    for symbol_name, refine_symbol in NODES_TO_REFINE.items():
        refine_symbol(root, symbol_name)
    version_constant = root.add_constant("__version__", "<unused>")
    version_constant._value_type = "str"

    """
    def redirectError(
        onError: Callable[[int, str, str, str, int], None] | None
    ) -> None: ...
    """
    root.add_function("redirectError", [
        FunctionNode.Arg(
            "onError",
            OptionalTypeNode(
                CallableTypeNode(
                    "ErrorCallback",
                    [
                        PrimitiveTypeNode.int_(),
                        PrimitiveTypeNode.str_(),
                        PrimitiveTypeNode.str_(),
                        PrimitiveTypeNode.str_(),
                        PrimitiveTypeNode.int_()
                    ]
                )
            )
        )
    ])


def make_optional_none_return(root_node: NamespaceNode,
                              function_symbol_name: SymbolName) -> None:
    """
    Make return type Optional[MatLike],
    for the functions that may return None.
    """
    function = find_function_node(root_node, function_symbol_name)
    for overload in function.overloads:
        if overload.return_type is not None:
            if not isinstance(overload.return_type.type_node, OptionalTypeNode):
                overload.return_type.type_node = OptionalTypeNode(
                    overload.return_type.type_node
                )

def export_matrix_type_constants(root: NamespaceNode) -> None:
    MAX_PREDEFINED_CHANNELS = 4

    depth_names = ("CV_8U", "CV_8S", "CV_16U", "CV_16S", "CV_32U", "CV_32S",
                   "CV_64U", "CV_64S", "CV_32F", "CV_64F", "CV_16F", "CV_16BF" "CV_Bool")
    for depth_value, depth_name in enumerate(depth_names):
        # Export depth constants
        root.add_constant(depth_name, str(depth_value))
        # Export predefined types
        for c in range(MAX_PREDEFINED_CHANNELS):
            root.add_constant(f"{depth_name}C{c + 1}",
                              f"{depth_value + 8 * c}")
        # Export type creation function
        root.add_function(
            f"{depth_name}C",
            (FunctionNode.Arg("channels", PrimitiveTypeNode.int_()), ),
            FunctionNode.RetType(PrimitiveTypeNode.int_())
        )
    # Export CV_MAKETYPE
    root.add_function(
        "CV_MAKETYPE",
        (FunctionNode.Arg("depth", PrimitiveTypeNode.int_()),
         FunctionNode.Arg("channels", PrimitiveTypeNode.int_())),
        FunctionNode.RetType(PrimitiveTypeNode.int_())
    )


def make_optional_arg(arg_name: str) -> Callable[[NamespaceNode, SymbolName], None]:
    def _make_optional_arg(root_node: NamespaceNode,
                           function_symbol_name: SymbolName) -> None:
        function = find_function_node(root_node, function_symbol_name)
        for overload in function.overloads:
            arg_idx = _find_argument_index(overload.arguments, arg_name)
            # Avoid multiplying optional qualification
            if isinstance(overload.arguments[arg_idx].type_node, OptionalTypeNode):
                continue

            overload.arguments[arg_idx].type_node = OptionalTypeNode(
                cast(TypeNode, overload.arguments[arg_idx].type_node)
            )

    return _make_optional_arg


def refine_cuda_module(root: NamespaceNode) -> None:
    def fix_cudaoptflow_enums_names() -> None:
        for class_name in ("NvidiaOpticalFlow_1_0", "NvidiaOpticalFlow_2_0"):
            if class_name not in cuda_root.classes:
                continue
            opt_flow_class = cuda_root.classes[class_name]
            _trim_class_name_from_argument_types(
                for_each_function_overload(opt_flow_class), class_name
            )

    def fix_namespace_usage_scope(cuda_ns: NamespaceNode) -> None:
        USED_TYPES = ("GpuMat", "Stream")

        def fix_type_usage(type_node: TypeNode) -> None:
            if isinstance(type_node, AggregatedTypeNode):
                for item in type_node.items:
                    fix_type_usage(item)
            if isinstance(type_node, ASTNodeTypeNode):
                if type_node._typename in USED_TYPES:
                    type_node._typename = f"cuda_{type_node._typename}"

        for overload in for_each_function_overload(cuda_ns):
            if overload.return_type is not None:
                fix_type_usage(overload.return_type.type_node)
            for type_node in [arg.type_node for arg in overload.arguments
                              if arg.type_node is not None]:
                fix_type_usage(type_node)

    if "cuda" not in root.namespaces:
        return
    cuda_root = root.namespaces["cuda"]
    fix_cudaoptflow_enums_names()
    for ns in [ns for ns_name, ns in root.namespaces.items()
               if ns_name.startswith("cuda")]:
        fix_namespace_usage_scope(ns)


def refine_highgui_module(root: NamespaceNode) -> None:
    # Check if library is built with enabled highgui module
    if "destroyAllWindows" not in root.functions:
        return
    """
    def createTrackbar(trackbarName: str,
                       windowName: str,
                       value: int,
                       count: int,
                       onChange: Callable[[int], None]) -> None: ...
    """
    root.add_function(
        "createTrackbar",
        [
            FunctionNode.Arg("trackbarName", PrimitiveTypeNode.str_()),
            FunctionNode.Arg("windowName", PrimitiveTypeNode.str_()),
            FunctionNode.Arg("value", PrimitiveTypeNode.int_()),
            FunctionNode.Arg("count", PrimitiveTypeNode.int_()),
            FunctionNode.Arg("onChange",
                             CallableTypeNode("TrackbarCallback",
                                              PrimitiveTypeNode.int_("int"))),
        ]
    )
    """
    def createButton(buttonName: str,
                     onChange: Callable[[tuple[int] | tuple[int, Any]], None],
                     userData: Any | None = ...,
                     buttonType: int = ...,
                     initialButtonState: int = ...) -> None: ...
    """
    root.add_function(
        "createButton",
        [
            FunctionNode.Arg("buttonName", PrimitiveTypeNode.str_()),
            FunctionNode.Arg(
                "onChange",
                CallableTypeNode(
                    "ButtonCallback",
                    UnionTypeNode(
                        "onButtonChangeCallbackData",
                        [
                            TupleTypeNode("onButtonChangeCallbackData",
                                          [PrimitiveTypeNode.int_(), ]),
                            TupleTypeNode("onButtonChangeCallbackData",
                                          [PrimitiveTypeNode.int_(),
                                           AnyTypeNode("void*")])
                        ]
                    )
                )),
            FunctionNode.Arg("userData",
                             OptionalTypeNode(AnyTypeNode("void*")),
                             default_value="None"),
            FunctionNode.Arg("buttonType", PrimitiveTypeNode.int_(),
                             default_value="0"),
            FunctionNode.Arg("initialButtonState", PrimitiveTypeNode.int_(),
                             default_value="0")
        ]
    )
    """
    def setMouseCallback(
        windowName: str,
        onMouse: Callback[[int, int, int, int, Any | None], None],
        param: Any | None = ...
    ) -> None: ...
    """
    root.add_function(
        "setMouseCallback",
        [
            FunctionNode.Arg("windowName", PrimitiveTypeNode.str_()),
            FunctionNode.Arg(
                "onMouse",
                CallableTypeNode("MouseCallback", [
                    PrimitiveTypeNode.int_(),
                    PrimitiveTypeNode.int_(),
                    PrimitiveTypeNode.int_(),
                    PrimitiveTypeNode.int_(),
                    OptionalTypeNode(AnyTypeNode("void*"))
                ])
            ),
            FunctionNode.Arg("param", OptionalTypeNode(AnyTypeNode("void*")),
                             default_value="None")
        ]
    )


def refine_dnn_module(root: NamespaceNode) -> None:
    if "dnn" not in root.namespaces:
        return
    dnn_module = root.namespaces["dnn"]

    """
    class LayerProtocol(Protocol):
        def __init__(
            self, params: dict[str, DictValue],
            blobs: typing.Sequence[cv2.typing.MatLike]
        ) -> None: ...

        def getMemoryShapes(
            self, inputs: typing.Sequence[typing.Sequence[int]]
        ) -> typing.Sequence[typing.Sequence[int]]: ...

        def forward(
            self, inputs: typing.Sequence[cv2.typing.MatLike]
        ) -> typing.Sequence[cv2.typing.MatLike]: ...
    """
    layer_proto = ProtocolClassNode("LayerProtocol", dnn_module)
    layer_proto.add_function(
        "__init__",
        arguments=[
            FunctionNode.Arg(
                "params",
                DictTypeNode(
                    "LayerParams", PrimitiveTypeNode.str_(),
                    create_type_node("cv::dnn::DictValue")
                )
            ),
            FunctionNode.Arg("blobs", create_type_node("vector<cv::Mat>"))
        ]
    )
    layer_proto.add_function(
        "getMemoryShapes",
        arguments=[
            FunctionNode.Arg("inputs",
                             create_type_node("vector<vector<int>>"))
        ],
        return_type=FunctionNode.RetType(
            create_type_node("vector<vector<int>>")
        )
    )
    layer_proto.add_function(
        "forward",
        arguments=[
            FunctionNode.Arg("inputs", create_type_node("vector<cv::Mat>"))
        ],
        return_type=FunctionNode.RetType(create_type_node("vector<cv::Mat>"))
    )

    """
    def dnn_registerLayer(layerTypeName: str,
                          layerClass: typing.Type[LayerProtocol]) -> None: ...
    """
    root.add_function(
        "dnn_registerLayer",
        arguments=[
            FunctionNode.Arg("layerTypeName", PrimitiveTypeNode.str_()),
            FunctionNode.Arg(
                "layerClass",
                ClassTypeNode(ASTNodeTypeNode(
                    layer_proto.export_name, f"dnn.{layer_proto.export_name}"
                ))
            )
        ]
    )

    """
    def dnn_unregisterLayer(layerTypeName: str) -> None: ...
    """
    root.add_function(
        "dnn_unregisterLayer",
        arguments=[
            FunctionNode.Arg("layerTypeName", PrimitiveTypeNode.str_())
        ]
    )


def _trim_class_name_from_argument_types(
    overloads: Iterable[FunctionNode.Overload],
    class_name: str
) -> None:
    separator = f"{class_name}_"
    for overload in overloads:
        for arg in [arg for arg in overload.arguments
                    if arg.type_node is not None]:
            ast_node = cast(ASTNodeTypeNode, arg.type_node)
            if class_name in ast_node.ctype_name:
                fixed_name = ast_node._typename.split(separator)[-1]
                ast_node._typename = fixed_name


def _find_argument_index(arguments: Sequence[FunctionNode.Arg],
                         name: str) -> int:
    for i, arg in enumerate(arguments):
        if arg.name == name:
            return i
    raise RuntimeError(
        f"Failed to find argument with name: '{name}' in {arguments}"
    )


NODES_TO_REFINE = {
    SymbolName(("cv", ), (), "resize"): make_optional_arg("dsize"),
    SymbolName(("cv", ), (), "calcHist"): make_optional_arg("mask"),
    SymbolName(("cv", ), (), "floodFill"): make_optional_arg("mask"),
    SymbolName(("cv", ), (), "imread"): make_optional_none_return,
    SymbolName(("cv", ), (), "imdecode"): make_optional_none_return,
}

ERROR_CLASS_PROPERTIES = (
    ClassProperty("code", PrimitiveTypeNode.int_(), False),
    ClassProperty("err", PrimitiveTypeNode.str_(), False),
    ClassProperty("file", PrimitiveTypeNode.str_(), False),
    ClassProperty("func", PrimitiveTypeNode.str_(), False),
    ClassProperty("line", PrimitiveTypeNode.int_(), False),
    ClassProperty("msg", PrimitiveTypeNode.str_(), False),
)
