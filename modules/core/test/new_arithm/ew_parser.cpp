// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Layer 4 implementation: see ew_parser.hpp.

#include "ew_parser.hpp"
#include "ew_compile.hpp"
#include "ew_exec.hpp"
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

namespace cv { namespace ew {

namespace {

// --- tokens ------------------------------------------------------------------------------
enum TokType { T_NUM, T_INPUT, T_IDENT, T_OP, T_LPAREN, T_RPAREN, T_COMMA, T_SEMI, T_ASSIGN, T_END };

struct Token
{
    TokType type = T_END;
    double num = 0;
    int input = 0;
    std::string text;     // identifier or operator spelling
};

// --- lexer -------------------------------------------------------------------------------
struct Lexer
{
    std::string_view s;
    size_t pos = 0;

    explicit Lexer(std::string_view src) : s(src) {}

    static bool isIdentStart(char c) { return std::isalpha((unsigned char)c) || c == '_'; }
    static bool isIdentChar(char c)  { return std::isalnum((unsigned char)c) || c == '_'; }

    Token next()
    {
        while (pos < s.size() && std::isspace((unsigned char)s[pos])) pos++;
        Token t;
        if (pos >= s.size()) { t.type = T_END; return t; }

        char c = s[pos];

        if (c == '{')                       // input placeholder {N}
        {
            pos++;
            size_t start = pos;
            while (pos < s.size() && s[pos] != '}') pos++;
            CV_Assert(pos < s.size() && "ew::expression: unterminated '{'");
            t.type = T_INPUT;
            t.input = std::atoi(std::string(s.substr(start, pos - start)).c_str());
            pos++;                          // consume '}'
            return t;
        }
        if (std::isdigit((unsigned char)c) || (c == '.' && pos + 1 < s.size() &&
                                               std::isdigit((unsigned char)s[pos + 1])))
        {
            char* end = nullptr;
            std::string num(s.substr(pos));
            t.type = T_NUM;
            t.num = std::strtod(num.c_str(), &end);
            pos += (size_t)(end - num.c_str());
            return t;
        }
        if (isIdentStart(c))
        {
            size_t start = pos;
            while (pos < s.size() && isIdentChar(s[pos])) pos++;
            t.type = T_IDENT;
            t.text.assign(s.substr(start, pos - start));
            return t;
        }
        switch (c)
        {
        case '(': pos++; t.type = T_LPAREN; return t;
        case ')': pos++; t.type = T_RPAREN; return t;
        case ',': pos++; t.type = T_COMMA;  return t;
        case ';': pos++; t.type = T_SEMI;   return t;
        }
        // multi/!single-char operators
        auto two = [&](const char* op) {
            return pos + 1 < s.size() && s[pos] == op[0] && s[pos + 1] == op[1];
        };
        t.type = T_OP;
        if (two("<=") || two(">=") || two("==") || two("!=")) { t.text.assign(s.substr(pos, 2)); pos += 2; return t; }
        if (c == '=') { pos++; t.type = T_ASSIGN; return t; }
        CV_Assert(std::strchr("+-*/<>&|^!", c) && "ew::expression: unexpected character");
        t.text.assign(1, c); pos++;
        return t;
    }
};

// --- operator / function tables ----------------------------------------------------------
static int binPrec(const std::string& op)
{
    if (op == "*" || op == "/") return 7;
    if (op == "+" || op == "-") return 6;
    if (op == "<" || op == "<=" || op == ">" || op == ">=") return 5;
    if (op == "==" || op == "!=") return 4;
    if (op == "&") return 3;
    if (op == "^") return 2;
    if (op == "|") return 1;
    return -1;
}

static ElemwiseOp binOp(const std::string& op)
{
    if (op == "+")  return OP_ADD;
    if (op == "-")  return OP_SUB;
    if (op == "*")  return OP_MUL;
    if (op == "/")  return OP_DIV;
    if (op == "<")  return OP_CMP_LT;
    if (op == "<=") return OP_CMP_LE;
    if (op == ">")  return OP_CMP_GT;
    if (op == ">=") return OP_CMP_GE;
    if (op == "==") return OP_CMP_EQ;
    if (op == "!=") return OP_CMP_NE;
    if (op == "&")  return OP_AND;
    if (op == "|")  return OP_OR;
    if (op == "^")  return OP_XOR;
    CV_Error(Error::StsParseError, "ew::expression: bad binary operator");
}

// type-cast function name -> depth, or -1 if not a type name
static int typeDepth(const std::string& name)
{
    if (name == "float")    return CV_32F;
    if (name == "double")   return CV_64F;
    if (name == "half" || name == "float16") return CV_16F;
    if (name == "bfloat16") return CV_16BF;
    if (name == "uint8")    return CV_8U;
    if (name == "int8")     return CV_8S;
    if (name == "uint16")   return CV_16U;
    if (name == "int16")    return CV_16S;
    if (name == "uint32")   return CV_32U;
    if (name == "int32")    return CV_32S;
    if (name == "uint64")   return CV_64U;
    if (name == "int64")    return CV_64S;
    return -1;
}

// element-wise op function name -> (op, arity), or arity 0 if unknown
static ElemwiseOp fnOp(const std::string& name, int& arity)
{
    struct E { const char* n; ElemwiseOp op; };
    static const E unary[]  = { {"abs",OP_ABS},{"sqrt",OP_SQRT},{"exp",OP_EXP},{"log",OP_LOG},
                                {"sin",OP_SIN},{"cos",OP_COS},{"tanh",OP_TANH},{"erf",OP_ERF},
                                {"relu",OP_RELU} };
    static const E binary[] = { {"max",OP_MAX},{"min",OP_MIN},{"pow",OP_POW},{"absdiff",OP_ABSDIFF} };
    static const E tern[]   = { {"clamp",OP_CLAMP},{"select",OP_SELECT} };
    for (const E& e : unary)  if (name == e.n) { arity = 1; return e.op; }
    for (const E& e : binary) if (name == e.n) { arity = 2; return e.op; }
    for (const E& e : tern)   if (name == e.n) { arity = 3; return e.op; }
    arity = 0; return OP_NOP;
}

// --- parser ------------------------------------------------------------------------------
struct Parser
{
    Lexer lex;
    Token cur;
    EwGraph& g;
    std::map<std::string, int> env;       // named temporaries

    Parser(std::string_view src, EwGraph& graph) : lex(src), g(graph) { cur = lex.next(); }

    void advance() { cur = lex.next(); }
    bool isOp(const char* op) const { return cur.type == T_OP && cur.text == op; }

    void expect(TokType t, const char* what)
    {
        CV_Assert(cur.type == t && what);
        advance();
    }

    int parsePrimary()
    {
        if (cur.type == T_NUM)   { int n = g.constant(Scalar(cur.num)); advance(); return n; }
        if (cur.type == T_INPUT) { int n = g.input(cur.input); advance(); return n; }
        if (cur.type == T_LPAREN){ advance(); int e = parseExpr(0); expect(T_RPAREN, "expected ')'"); return e; }
        if (cur.type == T_IDENT)
        {
            std::string name = cur.text; advance();
            if (cur.type != T_LPAREN)             // variable reference
            {
                auto it = env.find(name);
                CV_Assert(it != env.end() && "ew::expression: undefined name");
                return it->second;
            }
            advance();                            // consume '('
            std::vector<int> args;
            if (cur.type != T_RPAREN)
            {
                args.push_back(parseExpr(0));
                while (cur.type == T_COMMA) { advance(); args.push_back(parseExpr(0)); }
            }
            expect(T_RPAREN, "expected ')'");

            int td = typeDepth(name);
            if (td >= 0) { CV_Assert(args.size() == 1); return g.cast(args[0], td); }

            int arity = 0; ElemwiseOp op = fnOp(name, arity);
            CV_Assert(arity != 0 && "ew::expression: unknown function");
            CV_Assert((int)args.size() == arity && "ew::expression: wrong number of arguments");
            if (arity == 1) return g.unary(op, args[0]);
            if (arity == 2) return g.binary(op, args[0], args[1]);
            return g.ternary(op, args[0], args[1], args[2]);
        }
        CV_Error(Error::StsParseError, "ew::expression: expected a primary expression");
    }

    int parseUnary()
    {
        if (isOp("-") || isOp("!"))
        {
            std::string op = cur.text; advance();
            int operand = parseUnary();
            // constant-fold a leading sign so that "-1.5" does not need an OP_NEG kernel
            if (op == "-" && g.nodes[operand].kind == NODE_CONST)
            {
                Scalar v = g.nodes[operand].cval;
                return g.constant(Scalar(-v[0], -v[1], -v[2], -v[3]), g.nodes[operand].cchannels);
            }
            return g.unary(op == "-" ? OP_NEG : OP_NOT, operand);
        }
        return parsePrimary();
    }

    int parseExpr(int minPrec)
    {
        int left = parseUnary();
        while (cur.type == T_OP)
        {
            int p = binPrec(cur.text);
            if (p < minPrec) break;
            std::string op = cur.text; advance();
            int right = parseExpr(p + 1);         // left-associative
            left = g.binary(binOp(op), left, right);
        }
        return left;
    }

    // final result: a single expression, or a top-level tuple (e0, e1, ...).
    void parseResult()
    {
        if (cur.type == T_LPAREN)
        {
            Lexer save = lex; Token savedCur = cur;       // backtrack point
            advance();
            int e0 = parseExpr(0);
            if (cur.type == T_COMMA)
            {
                g.output(e0);
                while (cur.type == T_COMMA) { advance(); g.output(parseExpr(0)); }
                expect(T_RPAREN, "expected ')'");
                return;
            }
            lex = save; cur = savedCur;                   // not a tuple -> reparse as expr
        }
        g.output(parseExpr(0));
    }

    void parse()
    {
        while (true)
        {
            if (cur.type == T_IDENT)
            {
                Lexer save = lex; Token savedCur = cur;
                std::string name = cur.text; advance();
                if (cur.type == T_ASSIGN)
                {
                    advance();
                    env[name] = parseExpr(0);
                    expect(T_SEMI, "expected ';' after assignment");
                    continue;
                }
                lex = save; cur = savedCur;               // not an assignment
            }
            parseResult();
            if (cur.type == T_SEMI) advance();
            break;
        }
        CV_Assert(cur.type == T_END && "ew::expression: trailing tokens");
        CV_Assert(!g.outputs.empty() && "ew::expression: no result");
    }
};

} // anonymous namespace

void expression(std::string_view expr, InputArrayOfArrays _inputs, OutputArrayOfArrays _outputs)
{
    std::vector<Mat> inputs;
    _inputs.getMatVector(inputs);

    EwGraph g;
    g.ninputs = (int)inputs.size();
    Parser(expr, g).parse();

    std::vector<int> depths(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++) depths[i] = inputs[i].depth();

    EwProgram prog = compile(g, depths);

    std::vector<Mat> outs;
    exec(prog, inputs, outs);

    _outputs.create((int)outs.size(), 1, CV_8U, -1);     // size the output container
    for (size_t i = 0; i < outs.size(); i++)
        _outputs.getMatRef((int)i) = outs[i];
}

}} // namespace cv::ew
