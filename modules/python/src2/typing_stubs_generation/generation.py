__all__ = ("generate_typing_stubs", )

from io import StringIO
from pathlib import Path
import re
from typing import (Callable, NamedTuple, Union, Set, Dict,
                    Collection, Tuple, List)
import warnings

from .ast_utils import (get_enclosing_namespace,
                        get_enum_module_and_export_name,
                        for_each_function_overload,
                        for_each_class)

from .predefined_types import PREDEFINED_TYPES
from .api_refinement import apply_manual_api_refinement

from .nodes import (ASTNode, ASTNodeType, NamespaceNode, ClassNode,
                    FunctionNode, EnumerationNode, ConstantNode,
                    ProtocolClassNode)

from .nodes.type_node import (TypeNode, AliasTypeNode, AliasRefTypeNode,
                              AggregatedTypeNode, ASTNodeTypeNode,
                              ConditionalAliasTypeNode, PrimitiveTypeNode)


def generate_typing_stubs(root: NamespaceNode, output_path: Path):
    """Generates typing stubs for the AST with root `root` and outputs
    created files tree to directory pointed by `output_path`.

    Stubs generation consist from 4 steps:
        1. Reconstruction of AST tree for header parser output.
        2. "Lazy" AST nodes resolution (type nodes used as function arguments
            and return types). Resolution procedure attaches every "lazy"
            AST node to the corresponding node in the AST created during step 1.
        3. Generation of the typing module content. Typing module doesn't exist
           in library code, but is essential place to define aliases widely used
           in stub files.
        4. Generation of typing stubs from the reconstructed AST.
           Every namespace corresponds to a Python module with the same name.
           Generation procedure is recursive repetition of the following steps
           for each namespace (module):
                - Collect and write required imports for the module
                - Write all module constants stubs
                - Write all module enumerations stubs
                - Write all module classes stubs, preserving correct declaration
                  order, when base classes go before their derivatives.
                - Write all module functions stubs
                - Repeat steps above for nested namespaces

    Args:
        root (NamespaceNode): Root namespace node of the library AST.
        output_path (Path): Path to output directory.
    """
    # Perform special handling for function arguments that has some conventions
    # not expressed in their API e.g. optionality of mutually exclusive arguments
    # without default values:
    # ```cxx
    # cv::resize(cv::InputArray src, cv::OutputArray dst, cv::Size dsize,
    #       double fx = 0.0, double fy = 0.0, int interpolation);
    # ```
    # should accept `None` as `dsize`:
    # ```python
    # cv2.resize(image, dsize=None, fx=0.5, fy=0.5)
    # ```
    apply_manual_api_refinement(root)
    # Most of the time type nodes miss their full name (especially function
    # arguments and return types), so resolution should start from the narrowest
    # scope and gradually expanded.
    # Example:
    #   ```cpp
    #   namespace cv {
    #   enum AlgorithmType {
    #       // ...
    #   };
    #   namespace detail {
    #   struct Algorithm {
    #       static Ptr<Algorithm> create(AlgorithmType alg_type);
    #   };
    #   } // namespace detail
    #   } // namespace cv
    #   ```
    # To resolve `alg_type` argument of function `create` having `AlgorithmType`
    # type from above example the following steps are done:
    #    1. Try to resolve against `cv::detail::Algorithm` - fail
    #    2. Try to resolve against `cv::detail` - fail
    #    3. Try to resolve against `cv` - success
    # The whole process should fail !only! when all possible scopes are
    # checked and at least 1 node is still unresolved.
    root.resolve_type_nodes()
    _generate_typing_module(root, output_path)
    _populate_reexported_symbols(root)
    _generate_typing_stubs(root, output_path)


def _generate_typing_stubs(root: NamespaceNode, output_path: Path) -> None:
    output_path = Path(output_path) / root.export_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect all imports required for module items declaration
    required_imports = _collect_required_imports(root)

    output_stream = StringIO()

    # Add empty __all__ dunder on top of the module
    output_stream.write("__all__: list[str] = []\n\n")

    # Write required imports at the top of file
    _write_required_imports(required_imports, output_stream)

    _write_reexported_symbols_section(root, output_stream)

    # NOTE: Enumerations require special handling, because all enumeration
    # constants are exposed as module attributes
    has_enums = _generate_section_stub(
        StubSection("# Enumerations", ASTNodeType.Enumeration), root,
        output_stream, 0
    )
    # Collect all enums from class level and export them to module level
    for class_node in root.classes.values():
        if _generate_enums_from_classes_tree(class_node, output_stream,
                                             indent=0):
            has_enums = True
    # 2 empty lines between enum and classes definitions
    if has_enums:
        output_stream.write("\n")

    # Write the rest of module content - classes and functions
    for section in STUB_SECTIONS:
        _generate_section_stub(section, root, output_stream, 0)
    # Dump content to the output file
    (output_path / "__init__.pyi").write_text(output_stream.getvalue())
    # Process nested namespaces
    for ns in root.namespaces.values():
        _generate_typing_stubs(ns, output_path)


class StubSection(NamedTuple):
    name: str
    node_type: ASTNodeType


STUB_SECTIONS = (
    StubSection("# Constants", ASTNodeType.Constant),
    # Enumerations are skipped due to special handling rules
    # StubSection("# Enumerations", ASTNodeType.Enumeration),
    StubSection("# Classes", ASTNodeType.Class),
    StubSection("# Functions", ASTNodeType.Function)
)


def _generate_section_stub(section: StubSection, node: ASTNode,
                           output_stream: StringIO, indent: int) -> bool:
    """Generates stub for a single type of children nodes of the provided node.

    Args:
        section (StubSection): section identifier that carries section name and
            type its nodes.
        node (ASTNode): root node with children nodes used for
        output_stream (StringIO): Output stream for all nodes stubs related to
            the given section.
        indent (int): Indent used for each line written to `output_stream`.

    Returns:
        bool: `True` if section has a content, `False` otherwise.
    """
    if section.node_type not in node._children:
        return False

    children = node._children[section.node_type]
    if len(children) == 0:
        return False

    output_stream.write(" " * indent)
    output_stream.write(section.name)
    output_stream.write("\n")
    stub_generator = NODE_TYPE_TO_STUB_GENERATOR[section.node_type]
    children = filter(lambda c: c.is_exported, children.values())  # type: ignore
    if hasattr(section.node_type, "weight"):
        children = sorted(children, key=lambda child: getattr(child, "weight"))  # type: ignore
    for child in children:
        stub_generator(child, output_stream, indent)  # type: ignore
    output_stream.write("\n")
    return True


def _generate_class_stub(class_node: ClassNode, output_stream: StringIO,
                         indent: int = 0) -> None:
    """Generates stub for the provided class node.

    Rules:
    - Read/write properties are converted to object attributes.
    - Readonly properties are converted to functions decorated with `@property`.
    - When return type of static functions matches class name - these functions
      are treated as factory functions and annotated with `@classmethod`.
    - In contrast to implicit `this` argument in C++ methods, in Python all
      "normal" methods have explicit `self` as their first argument.
    - Body of empty classes is replaced with `...`

    Example:
    ```cpp
    struct Object : public BaseObject {
        struct InnerObject {
            int param;
            bool param2;

            float readonlyParam();
        };

        Object(int param, bool param2 = false);

        Object(InnerObject obj);

        static Object create();

    };
    ```
    becomes
    ```python
    class Object(BaseObject):
        class InnerObject:
            param: int
            param2: bool

            @property
            def readonlyParam() -> float: ...

        @typing.override
        def __init__(self, param: int, param2: bool = ...) -> None: ...

        @typing.override
        def __init__(self, obj: "Object.InnerObject") -> None: ...

        @classmethod
        def create(cls) -> Object: ...
    ```

    Args:
        class_node (ClassNode): Class node to generate stub entry for.
        output_stream (StringIO): Output stream for class stub.
        indent (int, optional): Indent used for each line written to
            `output_stream`. Defaults to 0.
    """

    class_module = get_enclosing_namespace(class_node)
    class_module_name = class_module.full_export_name

    if len(class_node.bases) > 0:
        bases = []
        for base in class_node.bases:
            base_module = get_enclosing_namespace(base)  # type: ignore
            if base_module != class_module:
                bases.append(base.full_export_name)
            else:
                bases.append(base.export_name)

        inheritance_str = f"({', '.join(bases)})"
    elif isinstance(class_node, ProtocolClassNode):
        inheritance_str = "(Protocol)"
    else:
        inheritance_str = ""

    output_stream.write(
        "{indent}class {name}{bases}:\n".format(
            indent=" " * indent,
            name=class_node.export_name,
            bases=inheritance_str
        )
    )
    has_content = len(class_node.properties) > 0

    # Processing class properties
    for property in class_node.properties:
        if property.is_readonly:
            template = "{indent}@property\n{indent}def {name}(self) -> {type}: ...\n"
        else:
            template = "{indent}{name}: {type}\n"

        output_stream.write(
            template.format(indent=" " * (indent + 4),
                            name=property.name,
                            type=property.relative_typename(class_module_name))
        )
    if len(class_node.properties) > 0:
        output_stream.write("\n")

    for section in STUB_SECTIONS:
        if _generate_section_stub(section, class_node,
                                  output_stream, indent + 4):
            has_content = True
    if not has_content:
        output_stream.write(" " * (indent + 4))
        output_stream.write("...\n\n")


def _generate_constant_stub(constant_node: ConstantNode,
                            output_stream: StringIO, indent: int = 0,
                            extra_export_prefix: str = "",
                            generate_uppercase_version: bool = True) -> Tuple[str, ...]:
    """Generates stub for the provided constant node.

    Args:
        constant_node (ConstantNode): Constant node to generate stub entry for.
        output_stream (StringIO): Output stream for constant stub.
        indent (int, optional): Indent used for each line written to
            `output_stream`. Defaults to 0.
        extra_export_prefix (str, optional): Extra prefix added to the export
            constant name. Defaults to empty string.
        generate_uppercase_version (bool, optional): Generate uppercase version
            alongside the normal one. Defaults to True.

    Returns:
        Tuple[str, ...]: exported constants names.
    """

    def write_constant_to_stream(export_name: str) -> None:
        output_stream.write(
            "{indent}{name}: {value_type}\n".format(
                name=export_name,
                value_type=constant_node.value_type,
                indent=" " * indent
            )
        )

    export_name = extra_export_prefix + constant_node.export_name
    write_constant_to_stream(export_name)
    if generate_uppercase_version:
        # Handle Python "magic" constants like __version__
        if re.match(r"^__.*__$", export_name) is not None:
            return export_name,

        uppercase_name = re.sub(r"([a-z])([A-Z])", r"\1_\2", export_name).upper()
        if export_name != uppercase_name:
            write_constant_to_stream(uppercase_name)
            return export_name, uppercase_name
    return export_name,


def _generate_enumeration_stub(enumeration_node: EnumerationNode,
                               output_stream: StringIO, indent: int = 0,
                               extra_export_prefix: str = "") -> None:
    """Generates stub for the provided enumeration node. In contrast to the
    Python `enum.Enum` class, C++ enumerations are exported as module-level
    (or class-level) constants.

    Example:
    ```cpp
    enum Flags {
        Flag1 = 0,
        Flag2 = 1,
        Flag3
    };
    ```
    becomes
    ```python
    Flag1: int
    Flag2: int
    Flag3: int
    Flags = int  # One of [Flag1, Flag2, Flag3]
    ```

    Unnamed enumerations don't export their names to Python:
    ```cpp
    enum {
        Flag1 = 0,
        Flag2 = 1
    };
    ```
    becomes
    ```python
    Flag1: int
    Flag2: int
    ```

    Scoped enumeration adds its name before each item name:
    ```cpp
    enum struct ScopedEnum {
        Flag1,
        Flag2
    };
    ```
    becomes
    ```python
    ScopedEnum_Flag1: int
    ScopedEnum_Flag2: int
    ScopedEnum = int # One of [ScopedEnum_Flag1, ScopedEnum_Flag2]
    ```

    Args:
        enumeration_node (EnumerationNode): Enumeration node to generate stub entry for.
        output_stream (StringIO): Output stream for enumeration stub.
        indent (int, optional): Indent used for each line written to `output_stream`.
            Defaults to 0.
        extra_export_prefix (str, optional) Extra prefix added to the export
            enumeration name. Defaults to empty string.
    """

    entries_extra_prefix = extra_export_prefix
    if enumeration_node.is_scoped:
        entries_extra_prefix += enumeration_node.export_name + "_"
    generated_constants_entries: List[str] = []
    for entry in enumeration_node.constants.values():
        generated_constants_entries.extend(
            _generate_constant_stub(entry, output_stream, indent, entries_extra_prefix)
        )
    # Unnamed enumerations are skipped as definition
    if enumeration_node.export_name.endswith("<unnamed>"):
        output_stream.write("\n")
        return
    output_stream.write(
        '{indent}{export_prefix}{name} = int\n{indent}"""One of [{entries}]"""\n\n'.format(
            export_prefix=extra_export_prefix,
            name=enumeration_node.export_name,
            entries=", ".join(generated_constants_entries),
            indent=" " * indent
        )
    )


def _generate_function_stub(function_node: FunctionNode,
                            output_stream: StringIO, indent: int = 0) -> None:
    """Generates stub entry for the provided function node. Function node can
    refer free function or class method.

    Args:
        function_node (FunctionNode): Function node to generate stub entry for.
        output_stream (StringIO): Output stream for function stub.
        indent (int, optional): Indent used for each line written to
            `output_stream`. Defaults to 0.
    """

    # Function is a stub without any arguments information
    if not function_node.overloads:
        warnings.warn(
            'Function node "{}" exported as "{}" has no overloads'.format(
                function_node.full_name, function_node.full_export_name
            )
        )
        return

    decorators = []
    if function_node.is_classmethod:
        decorators.append(" " * indent + "@classmethod")
    elif function_node.is_static:
        decorators.append(" " * indent + "@staticmethod")
    if len(function_node.overloads) > 1:
        decorators.append(" " * indent + "@_typing.overload")

    function_module = get_enclosing_namespace(function_node)
    function_module_name = function_module.full_export_name

    for overload in function_node.overloads:
        # Annotate every function argument
        annotated_args = []
        for arg in overload.arguments:
            annotated_arg = arg.name
            typename = arg.relative_typename(function_module_name)
            if typename is not None:
                annotated_arg += ": " + typename
            if arg.default_value is not None:
                annotated_arg += " = ..."
            annotated_args.append(annotated_arg)

        # And convert return type to the actual type
        if overload.return_type is not None:
            ret_type = overload.return_type.relative_typename(function_module_name)
        else:
            ret_type = "None"

        output_stream.write(
            "{decorators}"
            "{indent}def {name}({args}) -> {ret_type}: ...\n".format(
                decorators="\n".join(decorators) +
                "\n" if len(decorators) > 0 else "",
                name=function_node.export_name,
                args=", ".join(annotated_args),
                ret_type=ret_type,
                indent=" " * indent
            )
        )
    output_stream.write("\n")


def _generate_enums_from_classes_tree(class_node: ClassNode,
                                      output_stream: StringIO,
                                      indent: int = 0,
                                      class_name_prefix: str = "") -> bool:
    """Recursively generates class-level enumerations on the module level
    starting from the `class_node`.

    NOTE: This function is required, because all enumerations are exported as
    module-level constants.

    Example:
    ```cpp
    namespace cv {
    struct TermCriteria {
        enum Type {
            COUNT = 1,
            MAX_ITER = COUNT,
            EPS = 2
        };
    };
    }  // namespace cv
    ```
    is exported to `__init__.pyi` of `cv` module as as
    ```python
    TermCriteria_COUNT: int
    TermCriteria_MAX_ITER: int
    TermCriteria_EPS: int
    TermCriteria_Type = int  # One of [COUNT, MAX_ITER, EPS]
    ```

    Args:
        class_node (ClassNode): Class node to generate enumerations stubs for.
        output_stream (StringIO): Output stream for enumerations stub.
        indent (int, optional): Indent used for each line written to
            `output_stream`. Defaults to 0.
        class_name_prefix (str, optional): Prefix used for enumerations and
            constants names. Defaults to "".

    Returns:
        bool: `True` if classes tree declares at least 1 enum, `False` otherwise.
    """

    class_name_prefix = class_node.export_name + "_" + class_name_prefix
    has_content = len(class_node.enumerations) > 0
    for enum_node in class_node.enumerations.values():
        _generate_enumeration_stub(enum_node, output_stream, indent,
                                   class_name_prefix)
    for cls in class_node.classes.values():
        if _generate_enums_from_classes_tree(cls, output_stream, indent,
                                             class_name_prefix):
            has_content = True
    return has_content


def check_overload_presence(node: Union[NamespaceNode, ClassNode]) -> bool:
    """Checks that node has at least 1 function with overload.

    Args:
        node (Union[NamespaceNode, ClassNode]): Node to check for overload
            presence.

    Returns:
        bool: True if input node has at least 1 function with overload, False
            otherwise.
    """
    for func_node in node.functions.values():
        if len(func_node.overloads) > 1:
            return True
    return False


def _collect_required_imports(root: NamespaceNode) -> Collection[str]:
    """Collects all imports required for classes and functions typing stubs
    declarations.

    Args:
        root (NamespaceNode): Namespace node to collect imports for

    Returns:
        Collection[str]: Collection of unique `import smth` statements required
        for classes and function declarations of `root` node.
    """

    def _add_required_usage_imports(type_node: TypeNode, imports: Set[str]):
        for required_import in type_node.required_usage_imports:
            imports.add(required_import)

    required_imports: Set[str] = set()
    # Check if typing module is required due to @overload decorator usage
    # Looking for module-level function with at least 1 overload
    has_overload = check_overload_presence(root)
    # if there is no module-level functions with overload, check its presence
    # during class traversing, including their inner-classes
    has_protocol = False
    for cls in for_each_class(root):
        if not has_overload and check_overload_presence(cls):
            has_overload = True
            required_imports.add("import typing as _typing")
        # Add required imports for class properties
        for prop in cls.properties:
            _add_required_usage_imports(prop.type_node, required_imports)
        # Add required imports for class bases
        for base in cls.bases:
            base_namespace = get_enclosing_namespace(base)  # type: ignore
            if base_namespace != root:
                required_imports.add(
                    "import " + base_namespace.full_export_name
                )
        if isinstance(cls, ProtocolClassNode):
            has_protocol = True

    if has_overload:
        required_imports.add("import typing as _typing")
    # Importing modules required to resolve functions arguments
    for overload in for_each_function_overload(root):
        for arg in filter(lambda a: a.type_node is not None,
                          overload.arguments):
            _add_required_usage_imports(arg.type_node, required_imports)  # type: ignore
        if overload.return_type is not None:
            _add_required_usage_imports(overload.return_type.type_node,
                                        required_imports)

    root_import = "import " + root.full_export_name
    if root_import in required_imports:
        required_imports.remove(root_import)

    if has_protocol:
        required_imports.add("import sys")
    ordered_required_imports = sorted(required_imports)

    # Protocol import always goes as last import statement
    if has_protocol:
        ordered_required_imports.append(
            """if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol"""
        )

    return ordered_required_imports


def _populate_reexported_symbols(root: NamespaceNode) -> None:
    # Re-export all submodules to allow referencing symbols in submodules
    # without submodule import. Example:
    # `cv2.aruco.ArucoDetector` should be accessible without `import cv2.aruco`
    def _reexport_submodule(ns: NamespaceNode) -> None:
        for submodule in ns.namespaces.values():
            ns.reexported_submodules.append(submodule.export_name)
            _reexport_submodule(submodule)

    _reexport_submodule(root)

    root.reexported_submodules.append("typing")

    # Special cases, symbols defined in possible pure Python submodules
    # should be
    root.reexported_submodules_symbols["mat_wrapper"].append("Mat")


def _write_reexported_symbols_section(module: NamespaceNode,
                                      output_stream: StringIO) -> None:
    """Write re-export section for the given module.

    Re-export statements have from `from module_name import smth as smth`.
    Example:
    ```python
    from cv2 import aruco as aruco
    from cv2 import cuda as cuda
    from cv2 import ml as ml
    from cv2.mat_wrapper import Mat as Mat
    ```

    Args:
        module (NamespaceNode): Module with re-exported symbols.
        output_stream (StringIO): Output stream for re-export statements.
    """

    parent_name = module.full_export_name
    for submodule in sorted(module.reexported_submodules):
        output_stream.write(
            "from {0} import {1} as {1}\n".format(parent_name, submodule)
        )

    for submodule, symbols in sorted(module.reexported_submodules_symbols.items(),
                                     key=lambda kv: kv[0]):
        for symbol in symbols:
            output_stream.write(
                "from {0}.{1} import {2} as {2}\n".format(
                    parent_name, submodule, symbol
                )
            )

    if len(module.reexported_submodules) or \
            len(module.reexported_submodules_symbols):
        output_stream.write("\n\n")


def _write_required_imports(required_imports: Collection[str],
                            output_stream: StringIO) -> None:
    """Writes all entries of `required_imports` to the `output_stream`.

    Args:
        required_imports (Collection[str]): Imports to write into the output
            stream.
        output_stream (StringIO): Output stream for import statements.
    """

    for required_import in required_imports:
        output_stream.write(required_import)
        output_stream.write("\n")
    if len(required_imports):
        output_stream.write("\n\n")


def _generate_typing_module(root: NamespaceNode, output_path: Path) -> None:
    """Generates stub file for typings module.
    Actual module doesn't exist, but it is an appropriate place to define
    all widely-used aliases.

    Args:
        root (NamespaceNode): AST root node used for type nodes resolution.
        output_path (Path): Path to typing module directory, where __init__.pyi
            will be written.
    """
    def register_alias_links_from_aggregated_type(type_node: TypeNode) -> None:
        assert isinstance(type_node, AggregatedTypeNode), \
            f"Provided type node '{type_node.ctype_name}' is not an aggregated type"

        for item in filter(lambda i: isinstance(i, AliasRefTypeNode), type_node):
            type_node = PREDEFINED_TYPES[item.ctype_name]
            if isinstance(type_node, AliasTypeNode):
                register_alias(type_node)
            elif isinstance(type_node, ConditionalAliasTypeNode):
                conditional_type_nodes[type_node.ctype_name] = type_node

    def create_alias_for_enum_node(enum_node_alias: AliasTypeNode) -> ConditionalAliasTypeNode:
        """Create conditional int alias corresponding to the given enum node.

        Args:
            enum_node (AliasTypeNode): Enumeration node to create conditional
                int alias for.

        Returns:
            ConditionalAliasTypeNode: conditional int alias node with same
                export name as enum.
        """
        enum_node = enum_node_alias.ast_node
        assert enum_node.node_type == ASTNodeType.Enumeration, \
            f"{enum_node} has wrong node type. Expected type: Enumeration."

        enum_export_name, enum_module_name = get_enum_module_and_export_name(
            enum_node
        )
        return ConditionalAliasTypeNode(
            enum_export_name,
            "_typing.TYPE_CHECKING",
            positive_branch_type=enum_node_alias,
            negative_branch_type=PrimitiveTypeNode.int_(enum_export_name),
            condition_required_imports=("import typing as _typing", )
        )

    def register_alias(alias_node: AliasTypeNode) -> None:
        typename = alias_node.typename
        # Check if alias is already registered
        if typename in aliases:
            return

        # Collect required imports for alias definition
        for required_import in alias_node.required_definition_imports:
            required_imports.add(required_import)

        if isinstance(alias_node.value, AggregatedTypeNode):
            # Check if collection contains a link to another alias
            register_alias_links_from_aggregated_type(alias_node.value)

            # Remove references to alias nodes
            for i, item in enumerate(alias_node.value.items):
                # Process enumerations only
                if not isinstance(item, ASTNodeTypeNode) or item.ast_node is None:
                    continue
                if item.ast_node.node_type != ASTNodeType.Enumeration:
                    continue
                enum_node = create_alias_for_enum_node(item)
                alias_node.value.items[i] = enum_node
                conditional_type_nodes[enum_node.ctype_name] = enum_node

        if isinstance(alias_node.value, ASTNodeTypeNode) \
                and alias_node.value.ast_node == ASTNodeType.Enumeration:
            enum_node = create_alias_for_enum_node(alias_node.ast_node)
            conditional_type_nodes[enum_node.ctype_name] = enum_node
            return

        # Strip module prefix from aliased types
        aliases[typename] = alias_node.value.full_typename.replace(
            root.export_name + ".typing.", ""
        )
        if alias_node.doc is not None:
            aliases[typename] += f'\n"""{alias_node.doc}"""'

    output_path = Path(output_path) / root.export_name / "typing"
    output_path.mkdir(parents=True, exist_ok=True)

    required_imports: Set[str] = set()
    aliases: Dict[str, str] = {}
    conditional_type_nodes: Dict[str, ConditionalAliasTypeNode] = {}

    # Resolve each node and register aliases
    TypeNode.compatible_to_runtime_usage = True
    for node in PREDEFINED_TYPES.values():
        node.resolve(root)
        if isinstance(node, AliasTypeNode):
            register_alias(node)
        elif isinstance(node, ConditionalAliasTypeNode):
            conditional_type_nodes[node.ctype_name] = node

    for node in conditional_type_nodes.values():
        for required_import in node.required_definition_imports:
            required_imports.add(required_import)

    output_stream = StringIO()
    output_stream.write("__all__ = [\n")
    for alias_name in aliases:
        output_stream.write(f'    "{alias_name}",\n')
    output_stream.write("]\n\n")

    _write_required_imports(required_imports, output_stream)

    # Add type checking time definitions as generated __init__.py content
    for _, type_node in conditional_type_nodes.items():
        output_stream.write(f"if {type_node.condition}:\n    ")
        output_stream.write(f"{type_node.typename} = {type_node.positive_branch_type.full_typename}\nelse:\n")
        output_stream.write(f"    {type_node.typename} = {type_node.negative_branch_type.full_typename}\n\n\n")

    for alias_name, alias_type in aliases.items():
        output_stream.write(f"{alias_name} = {alias_type}\n")

    TypeNode.compatible_to_runtime_usage = False
    (output_path / "__init__.py").write_text(output_stream.getvalue())


StubGenerator = Callable[[ASTNode, StringIO, int], None]


NODE_TYPE_TO_STUB_GENERATOR = {
    ASTNodeType.Class: _generate_class_stub,
    ASTNodeType.Constant: _generate_constant_stub,
    ASTNodeType.Enumeration: _generate_enumeration_stub,
    ASTNodeType.Function: _generate_function_stub
}
