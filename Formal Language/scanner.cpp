#include <iostream>
#include <string>
#include <unordered_map>
#include <cctype>
using namespace std;

enum TokenType {
    END_OF_FILE,
    NUM,        // 数字
    ID,         // 标识符
    SCO,       // ;
    CMA,       // ,
    ASSIGN,     // =
    EQ,         // ==
    ADD,       // +
    MIN,      // -
    MUL,   // *
    DIV,     // /
    LBK,        // [
    RBK,        // ]
    LPA,     // (
    RPA,     // )
    LBR,     // {
    RBR,     // }
    LT,         // <
    LE,         // <=
    MT,         // >
    ME,         // >=
    IF,         // if
    ELSE,       // else
    WHILE,      // while
    RETURN,     // return
    INT,        // int
    VOID,       // void
    PRINT,      // print
    SDI,        // 带符号数字
};

struct Token 
{
    TokenType type; // 词法单元
    string lexeme; // 具体名字
    int numValue;  // 具体数值
    Token(TokenType t, const string& l, int v = 0)
        : type(t), lexeme(l), numValue(v) {}
};

class Lexer 
{
private:
    string input;
    size_t pos = 0;

    char currentChar() { return (pos < input.size()) ? input[pos] : '\0'; }
    char peekNextChar() { return (pos+1 < input.size()) ? input[pos+1] : '\0'; } // 用于区分 +5 和 + 5，正确拆分带符号数字

    void advance() { if (pos < input.size()) pos++; } // 迭代函数，移动当前位置
    
    void skipWhitespace() // 跳过空格
    {
        while (isspace(currentChar())) advance();
    }

    Token parseNumber() // 对数字类词法单元的处理
    {
        string numStr;
        while (isdigit(currentChar())) 
        {
            numStr += currentChar();
            advance();
        }
        return Token(NUM, numStr, stoi(numStr));
    }

    Token parseSignedNumber() // 带符号数字的处理
    {
        string numStr(1, currentChar());
        advance(); // 吃掉符号
        
        while (isdigit(currentChar())) 
        {
            numStr += currentChar();
            advance();
        }
        return Token(SDI, numStr, stoi(numStr));
    }

    Token parseIdentifier() // 对字符类词法单元的处理
    {
        string idStr;
        while (isalnum(currentChar()) || currentChar() == '_') 
        {
            idStr += currentChar();
            advance();
        }
        
        static const unordered_map<string, TokenType> keywords = // 关键词映射
        {
            {"if", IF}, 
            {"else", ELSE},
            {"while", WHILE}, 
            {"return", RETURN},
            {"print", PRINT},
            {"int", INT},
            {"void", VOID}
        };
        
        auto it = keywords.find(idStr);
        return it != keywords.end() ? 
            Token(it->second, idStr) : Token(ID, idStr); // 如果该词在关键词库中，则为关键词，否则为ID
    }

    Token parseAssignOrEq() // 区分 = 和 ==，作业没有要求，顺手写的
    {
        advance();
        if (currentChar() == '=') 
        {
            advance();
            return Token(EQ, "==");
        }
        return Token(ASSIGN, "=");
    }

    Token parseLessOrLE() // 区分 < 和 <=
    {
        advance();
        if (currentChar() == '=') 
        {
            advance();
            return Token(LE, "<=");
        }
        return Token(LT, "<");
    }

    Token parseMoreOrME() // 区分 > 和 >=
    {
        advance();
        if (currentChar() == '=') 
        {
            advance();
            return Token(ME, ">=");
        }
        return Token(MT, ">");
    }

public:
    Lexer(const string& s) : input(s) {}

    Token getNextToken() 
    {
        while (currentChar() != '\0') 
        {
            if (isspace(currentChar())) 
            {
                skipWhitespace();
                continue;
            }

            if ((currentChar() == '+' || currentChar() == '-') && isdigit(peekNextChar())) // 处理带符号数字
            {
                return parseSignedNumber();
            }

            if (isdigit(currentChar())) 
            {
                return parseNumber();
            }

            if (isalpha(currentChar()) || currentChar() == '_') 
            {
                return parseIdentifier();
            }

            switch (currentChar()) 
            {
                case ';': advance(); return Token(SCO, ";");
                case ',': advance(); return Token(CMA, ",");
                case '=': return parseAssignOrEq();
                case '+': advance(); return Token(ADD, "+");
                case '-': advance(); return Token(MIN, "-");
                case '*': advance(); return Token(MUL, "*");
                case '/': advance(); return Token(DIV, "/");
                case '[': advance(); return Token(LBK, "[");
                case ']': advance(); return Token(RBK, "]");
                case '(': advance(); return Token(LPA, "(");
                case ')': advance(); return Token(RPA, ")");
                case '{': advance(); return Token(LBR, "{");
                case '}': advance(); return Token(RBR, "}");
                case '<': return parseLessOrLE();
                case '>': return parseMoreOrME();
                default: throw runtime_error("Invalid char: " + string(1, currentChar()));
            }
        }
        return Token(END_OF_FILE, "");
    }
};

int main() 
{
    string codeArr[] = {
        "int raw(int x;){",
        "    y=x+5;",
        "    return y};",
        "void foo(int y;){",
        "    int z;",
        "    void bar(int x; int soo();){",
        "        if(x>3) bar(x/3, soo(),)", 
        "        else z = soo(x);",
        "        print z};",
        "    bar(y, raw(),)};",
        "foo(6,)"
    };
    for(int i = 0; i < 11; i++)
    {
        Lexer lexer(codeArr[i]);
        for (Token tok = lexer.getNextToken(); tok.type != END_OF_FILE; tok = lexer.getNextToken()) 
        {
            cout << "Token: ";
            switch (tok.type) 
            {
                case NUM:    cout << "(NUM, " << tok.numValue << ")"; break;
                case ID:     cout << "(ID, " << tok.lexeme << ")"; break;
                case SCO:    cout << "(SCO, ;)"; break;
                case CMA:    cout << "(CMA, ,)"; break;
                case ASSIGN: cout << "(ASG, =)"; break;
                case EQ:     cout << "(EQ, ==)"; break;
                case LT:     cout << "(ROF, " << tok.lexeme << ")"; break;
                case LE:     cout << "(ROF, " << tok.lexeme << ")"; break;
                case MT:     cout << "(ROF, " << tok.lexeme << ")"; break;
                case ME:     cout << "(ROF, " << tok.lexeme << ")"; break;
                case IF:     cout << "(IF, if)"; break;
                case ELSE:   cout << "(ELSE, else)"; break;
                case VOID:   cout << "(VOID, void)"; break;
                case INT:    cout << "(INT, int)"; break;
                case LBK:    cout << "(LBK, [)"; break;
                case RBK:    cout << "(RBK, ])"; break;
                case LPA:    cout << "(LPA, ()"; break;
                case RPA:    cout << "(RPA, ))"; break;
                case LBR:    cout << "(LBR, {)"; break;
                case RBR:    cout << "(RBR, })"; break;
                case DIV:    cout << "(DIV, /)"; break;
                case ADD:    cout << "(ADD, +)"; break;
                case PRINT:  cout << "(PRINT, print)"; break;
                case RETURN: cout << "(RETURN, return)"; break;
                case SDI:    cout << "(NUM, " << tok.lexeme << ")"; break;
                default: cout << tok.lexeme;
            }
            cout << endl;
        }
    }
}