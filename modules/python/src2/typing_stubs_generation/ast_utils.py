from typing import (NamedTuple, Sequence, Tuple, Union, List,
                    Dict, Callable, Optional)
import keyword

from .nodes import (ASTNode, NamespaceNode, ClassNode, FunctionNode,
                    EnumerationNode, ClassProperty, OptionalTypeNode,
                    TupleTypeNode)

from .types_conversion import create_type_node


class ScopeNotFoundError(Exception):
    pass


class SymbolNotFoundError(Exception):
    pass


class SymbolName(NamedTuple):
    namespaces: Tuple[str, ...]
    classes: Tuple[str, ...]
    name: str

    def __str__(self) -> str:
        return '(namespace="{}", classes="{}", name="{}")'.format(
            '::'.join(self.namespaces),
            '::'.join(self.classes),
            self.name
        )

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def parse(cls, full_symbol_name: str,
              known_namespaces: Sequence[str],
              symbol_parts_delimiter: str = '.') -> "SymbolName":
        """Performs contextual symbol name parsing into namespaces, classes
        and "bare" symbol name.

        Args:
            full_symbol_name (str): Input string to parse symbol name from.
            known_namespaces (Sequence[str]): Collection of namespace that was
                met during C++ headers parsing.
            symbol_parts_delimiter (str, optional): Delimiter string used to
                split `full_symbol_name` string into chunks. Defaults to '.'.

        Returns:
            SymbolName: Parsed symbol name structure.

        >>> SymbolName.parse('cv.ns.Feature', ('cv', 'cv.ns'))
        (namespace="cv::ns", classes="", name="Feature")

        >>> SymbolName.parse('cv.ns.Feature', ())
        (namespace="", classes="cv::ns", name="Feature")

        >>> SymbolName.parse('cv.ns.Feature.Params', ('cv', 'cv.ns'))
        (namespace="cv::ns", classes="Feature", name="Params")

        >>> SymbolName.parse('cv::ns::Feature::Params::serialize',
        ...                  known_namespaces=('cv', 'cv.ns'),
        ...                  symbol_parts_delimiter='::')
        (namespace="cv::ns", classes="Feature::Params", name="serialize")
        """

        chunks = full_symbol_name.split(symbol_parts_delimiter)
        namespaces, name = chunks[:-1], chunks[-1]
        classes: List[str] = []
        while len(namespaces) > 0 and '.'.join(namespaces) not in known_namespaces:
            classes.insert(0, namespaces.pop())
        return SymbolName(tuple(namespaces), tuple(classes), name)


def find_scope(root: NamespaceNode, symbol_name: SymbolName,
               create_missing_namespaces: bool = True) -> Union[NamespaceNode, ClassNode]:
    """Traverses down nodes hierarchy to the direct parent of the node referred
    by `symbol_name`.

    Args:
        root (NamespaceNode): Root node of the hierarchy.
        symbol_name (SymbolName): Full symbol name to find scope for.
        create_missing_namespaces (bool, optional): Set to True to create missing
            namespaces while traversing the hierarchy. Defaults to True.

    Raises:
        ScopeNotFoundError: If direct parent for the node referred by `symbol_name`
            can't be found e.g. one of classes doesn't exist.

    Returns:
        Union[NamespaceNode, ClassNode]: Direct parent for the node referred by
            `symbol_name`.

    >>> root = NamespaceNode('cv')
    >>> algorithm_node = root.add_class('Algorithm')
    >>> find_scope(root, SymbolName(('cv', ), ('Algorithm',), 'Params')) == algorithm_node
    True

    >>> root = NamespaceNode('cv')
    >>> scope = find_scope(root, SymbolName(('cv', 'gapi', 'detail'), (), 'function'))
    >>> scope.full_export_name
    'cv.gapi.detail'

    >>> root = NamespaceNode('cv')
    >>> scope = find_scope(root, SymbolName(('cv', 'gapi'), ('GOpaque',), 'function'))
    Traceback (most recent call last):
    ...
    ast_utils.ScopeNotFoundError: Can't find a scope for 'function', with \
'(namespace="cv::gapi", classes="GOpaque", name="function")', \
because 'GOpaque' class is not registered yet
    """
    assert isinstance(root, NamespaceNode), \
        'Wrong hierarchy root type: {}'.format(type(root))

    assert symbol_name.namespaces[0] == root.name, \
        "Trying to find scope for '{}' with root namespace different from: '{}'".format(
            symbol_name, root.name
    )

    scope: Union[NamespaceNode, ClassNode] = root
    for namespace in symbol_name.namespaces[1:]:
        if namespace not in scope.namespaces:  # type: ignore
            if not create_missing_namespaces:
                raise ScopeNotFoundError(
                    "Can't find a scope for '{}', with '{}', because namespace"
                    " '{}' is not created yet and `create_missing_namespaces`"
                    " flag is set to False".format(
                        symbol_name.name, symbol_name, namespace
                    )
                )
            scope = scope.add_namespace(namespace)  # type: ignore
        else:
            scope = scope.namespaces[namespace]  # type: ignore
    for class_name in symbol_name.classes:
        if class_name not in scope.classes:
            raise ScopeNotFoundError(
                "Can't find a scope for '{}', with '{}', because '{}' "
                "class is not registered yet".format(
                    symbol_name.name, symbol_name, class_name
                )
            )
        scope = scope.classes[class_name]
    return scope


def find_class_node(root: NamespaceNode, class_symbol: SymbolName,
                    create_missing_namespaces: bool = False) -> ClassNode:
    scope = find_scope(root, class_symbol, create_missing_namespaces)
    if class_symbol.name not in scope.classes:
        raise SymbolNotFoundError(
            "Can't find {} in its scope".format(class_symbol)
        )
    return scope.classes[class_symbol.name]


def find_function_node(root: NamespaceNode, function_symbol: SymbolName,
                       create_missing_namespaces: bool = False) -> FunctionNode:
    scope = find_scope(root, function_symbol, create_missing_namespaces)
    if function_symbol.name not in scope.functions:
        raise SymbolNotFoundError(
            "Can't find {} in its scope".format(function_symbol)
        )
    return scope.functions[function_symbol.name]


def create_function_node_in_scope(scope: Union[NamespaceNode, ClassNode],
                                  func_info) -> FunctionNode:
    def prepare_overload_arguments_and_return_type(variant):
        arguments = []  # type: list[FunctionNode.Arg]
        # Enumerate is requried, because `argno` in `variant.py_arglist`
        # refers to position of argument in C++ function interface,
        # but `variant.py_noptargs` refers to position in `py_arglist`
        for i, (_, argno) in enumerate(variant.py_arglist):
            arg_info = variant.args[argno]
            type_node = create_type_node(arg_info.tp)
            default_value = None
            if len(arg_info.defval):
                default_value = arg_info.defval
            # If argument is optional and can be None - make its type optional
            if variant.is_arg_optional(i):
                # NOTE: should UMat be always mandatory for better type hints?
                # otherwise overload won't be selected e.g. VideoCapture.read()
                if arg_info.py_outputarg:
                    type_node = OptionalTypeNode(type_node)
                    default_value = "None"
                elif arg_info.isbig() and "None" not in type_node.typename:
                    # but avoid duplication of the optioness
                    type_node = OptionalTypeNode(type_node)
            arguments.append(
                FunctionNode.Arg(arg_info.export_name, type_node=type_node,
                                 default_value=default_value)
            )
        if func_info.isconstructor:
            return arguments, None

        # Function has more than 1 output argument, so its return type is a tuple
        if len(variant.py_outlist) > 1:
            ret_types = []
            # Actual returned value of the function goes first
            if variant.py_outlist[0][1] == -1:
                ret_types.append(create_type_node(variant.rettype))
                outlist = variant.py_outlist[1:]
            else:
                outlist = variant.py_outlist
            for _, argno in outlist:
                assert argno >= 0, \
                    "Logic Error! Outlist contains function return type: {}".format(
                        outlist
                    )

                ret_types.append(create_type_node(variant.args[argno].tp))

            return arguments, FunctionNode.RetType(
                TupleTypeNode("return_type", ret_types)
            )
        # Function with 1 output argument in Python
        if len(variant.py_outlist) == 1:
            # Can be represented as a function with a non-void return type in C++
            if variant.rettype:
                return arguments, FunctionNode.RetType(
                    create_type_node(variant.rettype)
                )
            # or a function with void return type and output argument type
            # such non-const reference
            ret_type = variant.args[variant.py_outlist[0][1]].tp
            return arguments, FunctionNode.RetType(
                create_type_node(ret_type)
            )
        # Function without output types returns None in Python
        return arguments, None

    function_node = FunctionNode(func_info.name)
    function_node.parent = scope
    if func_info.isconstructor:
        function_node.export_name = "__init__"
    for variant in func_info.variants:
        arguments, ret_type = prepare_overload_arguments_and_return_type(variant)
        if isinstance(scope, ClassNode):
            if func_info.is_static:
                if ret_type is not None and ret_type.typename.endswith(scope.name):
                    function_node.is_classmethod = True
                    arguments.insert(0, FunctionNode.Arg("cls"))
                else:
                    function_node.is_static = True
            else:
                arguments.insert(0, FunctionNode.Arg("self"))
        function_node.add_overload(arguments, ret_type)
    return function_node


def create_function_node(root: NamespaceNode, func_info) -> FunctionNode:
    func_symbol_name = SymbolName(
        func_info.namespace.split(".") if len(func_info.namespace) else (),
        func_info.classname.split(".") if len(func_info.classname) else (),
        func_info.name
    )
    return create_function_node_in_scope(find_scope(root, func_symbol_name),
                                         func_info)


def create_class_node_in_scope(scope: Union[NamespaceNode, ClassNode],
                               symbol_name: SymbolName,
                               class_info) -> ClassNode:
    properties = []
    for property in class_info.props:
        export_property_name = property.name
        if keyword.iskeyword(export_property_name):
            export_property_name += "_"
        properties.append(
            ClassProperty(
                name=export_property_name,
                type_node=create_type_node(property.tp),
                is_readonly=property.readonly
            )
        )
    class_node = scope.add_class(symbol_name.name,
                                 properties=properties)
    class_node.export_name = class_info.export_name
    if class_info.constructor is not None:
        create_function_node_in_scope(class_node, class_info.constructor)
    for method in class_info.methods.values():
        create_function_node_in_scope(class_node, method)
    return class_node


def create_class_node(root: NamespaceNode, class_info,
                      namespaces: Sequence[str]) -> ClassNode:
    symbol_name = SymbolName.parse(class_info.full_original_name, namespaces)
    scope = find_scope(root, symbol_name)
    return create_class_node_in_scope(scope, symbol_name, class_info)


def resolve_enum_scopes(root: NamespaceNode,
                        enums: Dict[SymbolName, EnumerationNode]):
    """Attaches all enumeration nodes to the appropriate classes and modules

    If classes containing enumeration can't be found in the AST - they will
    be created and marked as not exportable. This behavior is required to cover
    cases, when enumeration is defined in base class, but only its derivatives
    are used. Example:
        ```cpp
        class CV_EXPORTS TermCriteria {
        public:
        enum Type { /* ... */ };
        // ...
        };
        ```

    Args:
        root (NamespaceNode): root of the reconstructed AST
        enums (Dict[SymbolName, EnumerationNode]): Mapping between enumerations
            symbol names and corresponding nodes without parents.
    """

    for symbol_name, enum_node in enums.items():
        if symbol_name.classes:
            try:
                scope = find_scope(root, symbol_name)
            except ScopeNotFoundError:
                # Scope can't be found if enumeration is a part of class
                # that is not exported.
                # Create class node, but mark it as not exported
                for i, class_name in enumerate(symbol_name.classes):
                    scope = find_scope(root,
                                       SymbolName(symbol_name.namespaces,
                                                  classes=symbol_name.classes[:i],
                                                  name=class_name))
                    if class_name in scope.classes:
                        continue
                    class_node = scope.add_class(class_name)
                    class_node.is_exported = False
                scope = find_scope(root, symbol_name)
        else:
            scope = find_scope(root, symbol_name)
        enum_node.parent = scope


def get_enclosing_namespace(
    node: ASTNode,
    class_node_callback: Optional[Callable[[ClassNode], None]] = None
) -> NamespaceNode:
    """Traverses up nodes hierarchy to find closest enclosing namespace of the
    passed node

    Args:
        node (ASTNode): Node to find a namespace for.
        class_node_callback (Optional[Callable[[ClassNode], None]]): Optional
            callable object invoked for each traversed class node in bottom-up
            order. Defaults: None.

    Returns:
        NamespaceNode: Closest enclosing namespace of the provided node.

    Raises:
        AssertionError: if nodes hierarchy missing a namespace node.

    >>> root = NamespaceNode('cv')
    >>> feature_class = root.add_class("Feature")
    >>> get_enclosing_namespace(feature_class) == root
    True

    >>> root = NamespaceNode('cv')
    >>> feature_class = root.add_class("Feature")
    >>> feature_params_class = feature_class.add_class("Params")
    >>> serialize_params_func = feature_params_class.add_function("serialize")
    >>> get_enclosing_namespace(serialize_params_func) == root
    True

    >>> root = NamespaceNode('cv')
    >>> detail_ns = root.add_namespace('detail')
    >>> flags_enum = detail_ns.add_enumeration('Flags')
    >>> get_enclosing_namespace(flags_enum) == detail_ns
    True
    """
    parent_node = node.parent
    while not isinstance(parent_node, NamespaceNode):
        assert parent_node is not None, \
            "Can't find enclosing namespace for '{}' known as: '{}'".format(
                node.full_export_name, node.native_name
            )
        if class_node_callback:
            class_node_callback(parent_node)
        parent_node = parent_node.parent
    return parent_node


def get_enum_module_and_export_name(enum_node: EnumerationNode) -> Tuple[str, str]:
    """Get export name of the enum node with its module name.

    Note: Enumeration export names are prefixed with enclosing class names.

    Args:
        enum_node (EnumerationNode): Enumeration node to construct name for.

    Returns:
        Tuple[str, str]: a pair of enum export name and its full module name.
    """
    def update_full_export_name(class_node: ClassNode) -> None:
        nonlocal enum_export_name
        enum_export_name = class_node.export_name + "_" + enum_export_name

    enum_export_name = enum_node.export_name
    namespace_node = get_enclosing_namespace(enum_node, update_full_export_name)
    return enum_export_name, namespace_node.full_export_name


if __name__ == '__main__':
    import doctest
    doctest.testmod()
