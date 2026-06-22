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

static TOp binOp(const std::string& op)
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
static TOp fnOp(const std::string& name, int& arity)
{
    struct E { const char* n; TOp op; };
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
// Builds the program DIRECTLY into the TExpr (no intermediate graph), exactly like the hand
// builders: each parse step calls e.emit()/e.addConst()/e.output() and returns the arg SLOT
// holding its value. Input depths are known up front (from the input Mats), so every operand is
// typed as it is parsed - emit() infers result depths and inserts casts on the spot.
struct Parser
{
    Lexer lex;
    Token cur;
    TExpr& e;
    const int* inputSlot;                 // slot id of each external input (precreated)
    int ninputs;
    std::map<std::string, int> env;       // named temporaries -> slot

    Parser(std::string_view src, TExpr& expr, const int* islot, int nin)
        : lex(src), e(expr), inputSlot(islot), ninputs(nin) { cur = lex.next(); }

    void advance() { cur = lex.next(); }
    bool isOp(const char* op) const { return cur.type == T_OP && cur.text == op; }

    void expect(TokType t, const char* what)
    {
        CV_Assert(cur.type == t && what);
        advance();
    }

    // A flexible CONST slot holds a literal whose depth emit() picks per use.
    bool isFlexConst(int slot) const
    {
        return e.arginfo[slot].kind == TExpr::CONST && e.arginfo[slot].depth == EW_DEPTH_NONE;
    }

    int parsePrimary()
    {
        if (cur.type == T_NUM)   { int s = e.addConst(EW_DEPTH_NONE, Scalar(cur.num), 1); advance(); return s; }
        if (cur.type == T_INPUT) { int idx = cur.input; advance();
                                   CV_Assert(idx >= 0 && idx < ninputs && "ew::expression: input index out of range");
                                   return inputSlot[idx]; }
        if (cur.type == T_LPAREN){ advance(); int x = parseExpr(0); expect(T_RPAREN, "expected ')'"); return x; }
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
            if (td >= 0) { CV_Assert(args.size() == 1); return e.emit(OP_CAST, args.data(), 1, td); }

            int arity = 0; TOp op = fnOp(name, arity);
            CV_Assert(arity != 0 && "ew::expression: unknown function");
            CV_Assert((int)args.size() == arity && "ew::expression: wrong number of arguments");
            return e.emit(op, args.data(), arity);
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
            if (op == "-" && isFlexConst(operand))
            {
                const TExpr::Arg& a = e.arginfo[operand];
                return e.addConst(EW_DEPTH_NONE, Scalar(-a.cval[0], -a.cval[1], -a.cval[2], -a.cval[3]), a.channels);
            }
            int x = operand;
            return e.emit(op == "-" ? OP_NEG : OP_NOT, &x, 1);
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
            int args[2] = { left, right };
            left = e.emit(binOp(op), args, 2);
        }
        return left;
    }

    // final result: a single expression, or a top-level tuple (e0, e1, ...).
    void parseResult()
    {
        if (cur.type == T_LPAREN)
        {
            // backtrack point: save the lexer AND the program length, since the trial parse below
            // emits instructions/slots that must be rolled back if this turns out NOT to be a tuple.
            Lexer save = lex; Token savedCur = cur;
            size_t nInsn = e.prog.size(), nArg = e.arginfo.size(); int nTemp = e.ntemps;
            advance();
            int e0 = parseExpr(0);
            if (cur.type == T_COMMA)
            {
                e.output(e0);
                while (cur.type == T_COMMA) { advance(); e.output(parseExpr(0)); }
                expect(T_RPAREN, "expected ')'");
                return;
            }
            lex = save; cur = savedCur;                   // not a tuple -> undo the trial parse
            e.prog.resize(nInsn); e.arginfo.resize(nArg); e.ntemps = nTemp;
        }
        e.output(parseExpr(0));
    }

    void parse()
    {
        while (true)
        {
            if (cur.type == T_IDENT)
            {
                Lexer save = lex; Token savedCur = cur;   // lexer-only backtrack (nothing emitted yet)
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
        CV_Assert(e.noutputs > 0 && "ew::expression: no result");
    }
};

} // anonymous namespace

void expression(std::string_view expr, InputArrayOfArrays _inputs, OutputArrayOfArrays _outputs)
{
    CV_Assert(_inputs.kind() == _InputArray::STD_VECTOR_MAT);
    const std::vector<Mat>& inps = *(const std::vector<Mat>*)_inputs.getObj();
    const int ninputs = (int)inps.size();

    // Input depths are known now, so build a fully-typed program straight away: one INPUT slot per
    // input (slot index i carries input i's depth), then parse the expression into instructions.
    TExpr e;
    AutoBuffer<int> islot(std::max(ninputs, 1));
    for (int i = 0; i < ninputs; i++) islot[i] = e.addInput(inps[i].depth());

    Parser(expr, e, islot.data(), ninputs).parse();
    e.compile();

    auto kind = _outputs.kind();
    if (kind == _InputArray::STD_VECTOR_MAT) {
        std::vector<Mat>& outs = _outputs.getMatVecRef();
        outs.resize(e.noutputs);
        e.exec(inps.data(), outs.data());
    } else {
        CV_Error(Error::StsNotImplemented, "vector<Mat> is expected as output of expression");
    }
}

}} // namespace cv::ew
