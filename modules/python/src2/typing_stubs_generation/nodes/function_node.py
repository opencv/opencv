from typing import NamedTuple, Sequence, Type, Optional, Tuple, List

from .node import ASTNode, ASTNodeType
from .type_node import TypeNode, NoneTypeNode, TypeResolutionError


class FunctionNode(ASTNode):
    """Represents a function (or class method) in both C++ and Python.

    This class defines an overload set rather then function itself, because
    function without overloads is represented as FunctionNode with 1 overload.
    """
    class Arg:
        def __init__(self, name: str, type_node: Optional[TypeNode] = None,
                     default_value: Optional[str] = None) -> None:
            self.name = name
            self.type_node = type_node
            self.default_value = default_value

        @property
        def typename(self) -> Optional[str]:
            return getattr(self.type_node, "full_typename", None)

        def relative_typename(self, root: str) -> Optional[str]:
            if self.type_node is not None:
                return self.type_node.relative_typename(root)
            return None

        def __str__(self) -> str:
            return (
                f"Arg(name={self.name}, type_node={self.type_node},"
                f" default_value={self.default_value})"
            )

        def __repr__(self) -> str:
            return str(self)

    class RetType:
        def __init__(self, type_node: TypeNode = NoneTypeNode("void")) -> None:
            self.type_node = type_node

        @property
        def typename(self) -> str:
            return self.type_node.full_typename

        def relative_typename(self, root: str) -> Optional[str]:
            return self.type_node.relative_typename(root)

        def __str__(self) -> str:
            return f"RetType(type_node={self.type_node})"

        def __repr__(self) -> str:
            return str(self)

    class Overload(NamedTuple):
        arguments: Sequence["FunctionNode.Arg"] = ()
        return_type: Optional["FunctionNode.RetType"] = None

    def __init__(self, name: str,
                 arguments: Optional[Sequence["FunctionNode.Arg"]] = None,
                 return_type: Optional["FunctionNode.RetType"] = None,
                 is_static: bool = False,
                 is_classmethod: bool = False,
                 parent: Optional[ASTNode] = None,
                 export_name: Optional[str] = None) -> None:
        """Function node initializer

        Args:
            name (str): Name of the function overload set
            arguments (Optional[Sequence[FunctionNode.Arg]], optional): Function
                arguments. If this argument is None, then no overloads are
                added and node should be treated like a "function stub" rather
                than function. This might be helpful if there is a knowledge
                that function with the defined name exists, but information
                about its interface is not available at that moment.
                Defaults to None.
            return_type (Optional[FunctionNode.RetType], optional): Function
                return type. Defaults to None.
            is_static (bool, optional): Flag pointing that function is
                a static method of some class. Defaults to False.
            is_classmethod (bool, optional): Flag pointing that function is
                a class method of some class. Defaults to False.
            parent (Optional[ASTNode], optional): Parent ASTNode of the function.
                Can be class or namespace. Defaults to None.
            export_name (Optional[str], optional): Export name of the function.
                Defaults to None.
        """

        super().__init__(name, parent, export_name)
        self.overloads: List[FunctionNode.Overload] = []
        self.is_static = is_static
        self.is_classmethod = is_classmethod
        if arguments is not None:
            self.add_overload(arguments, return_type)

    @property
    def node_type(self) -> ASTNodeType:
        return ASTNodeType.Function

    @property
    def children_types(self) -> Tuple[Type[ASTNode], ...]:
        return ()

    def add_overload(self, arguments: Sequence["FunctionNode.Arg"] = (),
                     return_type: Optional["FunctionNode.RetType"] = None):
        self.overloads.append(FunctionNode.Overload(arguments, return_type))

    def resolve_type_nodes(self, root: ASTNode):
        """Resolves type nodes in all overloads against `root`

        Type resolution errors are postponed until all type nodes are examined.

        Args:
            root (ASTNode): Root of AST sub-tree used for type nodes resolution.
        """
        def has_unresolved_type_node(item) -> bool:
            return item.type_node is not None and not item.type_node.is_resolved

        errors = []
        for overload in self.overloads:
            for arg in filter(has_unresolved_type_node, overload.arguments):
                try:
                    arg.type_node.resolve(root)  # type: ignore
                except TypeResolutionError as e:
                    errors.append(
                        'Failed to resolve "{}" argument: {}'.format(arg.name, e)
                    )
            if overload.return_type is not None and \
                    has_unresolved_type_node(overload.return_type):
                try:
                    overload.return_type.type_node.resolve(root)
                except TypeResolutionError as e:
                    errors.append('Failed to resolve return type: {}'.format(e))
        if len(errors) > 0:
            raise TypeResolutionError(
                'Failed to resolve "{}" function against "{}". Errors: {}'.format(
                    self.full_export_name, root.full_export_name,
                    ", ".join("[{}]: {}".format(i, e) for i, e in enumerate(errors))
                )
            )
