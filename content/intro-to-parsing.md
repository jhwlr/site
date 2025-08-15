+++
title = "A brief introduction to parsing"
date = "2025-08-15"

[extra]
add_toc = true
+++

---

If you're reading this, you probably know how to program a computer. You write some text,
and use a special thing called a "compiler" (sometimes also "interpreter") to turn that
text into instructions for your computer[^1].

I want to attempt to explain[^10] a small part of
the process of transforming text into those instructions, usually referred to as "parsing",
and then implement a _parser_ in Rust.

## Parsing programming languages

As a programmer, you understand how to recognize different parts of a programming language.
It's similar to reading any other form of text. You read it one line at a time, or by scanning
for patterns. Each chunk is given some kind of meaning based on its contents. You intuitively
understand the _grammar_ of a programming language, quickly transforming it into semantics
in your head.

<div style="display:flex; justify-content:center; width: 100%">
<img style="filter:none" width="50%" src="/intro-to-parsing/tokens.svg">
</div>

But a computer doesn't know how to do that without your help. It's your job to break up the text into
smaller pieces, the individual _syntactic elements_ of a program. You also have to assign each piece
of syntax some sort of meaning. The program which does this is usually called a _parser_, and is
often only one _stage_ in a full [_compiler_](https://en.wikipedia.org/wiki/Compiler).

The parser reads _source code_, and transforms it into a _syntax tree_[^2]. Depending on what exactly
you're trying to do, this syntax tree can either be _concrete_ or _abstract_.

A concrete syntax tree (CST)
stores everything necessary to preserve the exact _appearance_ of a program: Parentheses, curly brackets,
keywords, and so on. An abstract syntax tree (AST) only stores whatever is necessary to preserve the program's
_meaning_, discarding everything else. We'll be using an AST[^3].

<div style="display:flex; justify-content:center; width: 100%">
<img style="filter:none" width="50%" src="/intro-to-parsing/AST.svg">
</div>

## A simple language

Our testbed for parsing will be <i>simp</i>le programming language, called `simp`[^4]. 

Here's a `simp`le program:

```rust
fn factorial(n) {
  if n <= 1 { n }
  else { n * factorial(n - 1) }
}

let fact_5 = factorial(5);
assert(fact_5 == 120, "5! must be 120");
```

It's so simple that it's _just barely_ useful. We have:
* Functions, and function calls <small>(including recursion!)</small>
* Variables
* Various operators
* Integers, strings

And not much more. The syntax is heavily inspired by Rust; it's vaguely C-like with curly
brackets and semicolons, is "expression-based"[^5], calls functions `fn`, and so on.

We have some fancy structured text. What now?

## Lexical analysis

To make the process of parsing easier, source text is typically first split up into a list of _tokens_,
referred to as "tokenization" or "lexing" (short for [lexical analysis](https://en.wikipedia.org/wiki/Lexical_analysis)).
It's not strictly necessary, but it is useful; `simp` does not use [significant indentation](https://en.wikipedia.org/wiki/Off-side_rule),
so using a lexer means we can discard all whitespace characters before we even start parsing.
Our parser becomes a lot simpler as a result.

Each token will need to store what _kind_ of token it is, and a _span_[^6]. We'll use the span to retrieve
the token's _lexeme_ -- its slice of the source code.

```rust
struct Token {
    kind: TokenKind,
    span: Span,
}
```

> Code snippets in this article are incomplete to preserve space;
> the full source code is [available on GitHub](https://github.com/jhwlr/simp). 

The core of our lexer is a loop, reading one _byte_[^8] at a time from our source text.
It'll also need to keep track of where in the source code it is.

```rust
fn lex(code: &str) -> Vec<Token> {
    let bytes = code.as_bytes();
    let mut pos = 0;
    while pos < bytes.len() {
        let start = pos; // token's starting position
        // ...
    }
}
```

Next we need to determine if we're looking at a character which is valid in this position.
Rust's pattern matching is convenient for this task.

To match one-byte tokens, such as `+` and `-`, we only need to match the first byte:

```rust
match bytes[pos] {
    b'+' => { /* ... */ }
    b'-' => { /* ... */ }
    // ...
}
```

For two-byte tokens, we'll need a nested match:

```rust
match bytes[pos] {
    // ...
    b'=' => if bytes.get(pos+1) == Some(&b'=') {
        // matched `==`
        pos += 2;
    } else {
        // matched only `=`
        pos += 1;
    }
}
```

Our final token "category" is an arbitrary-length sequence.
Our lexer is _greedy_, meaning it continues to "append" characters to the current token for
as long as it's a valid sequence. We'll use an inner loop for that:

```rust
match bytes[pos] {
    // ...

    // identifier
    b'_' | b'a'..=b'z' | b'A'..=b'Z' => {
        pos += 1;

        while let Some(byte) = bytes.get(pos)
            && let b'_' | b'a'..=b'z' | b'A'..=b'Z' = *byte
        {
            pos += 1;
        }

        // ...
    }
}
```

`simp` also has a few keywords. After we lex an identifier, we'll have to check if it's a keyword,
and output the token kind for that keyword instead.

Once we've "consumed" some valid sequence of characters, we can construct a token:

```rust
let end = pos;
Token {
    kind: /*...*/,
    span: Span::from(start..end),
}
```

## Lexical analysis, automated

While I'd be happy to write a lexer by hand, it tends to be a very error-prone, tedious task.
To make our job easier, we'll use a library called [Logos](https://github.com/maciejhirsz/logos).
It's easy to use, and generates _very_ fast lexers, the kind which would be very impractical
to write by hand.

I intentionally omitted the definition of our `TokenKind` enum until now. Here it is:

```rust
#[derive(Logos)]
#[logos(skip r"[ \t\n\r]+")]
enum TokenKind {
    #[token("fn")] KW_FN,
    #[token("if")] KW_IF,
    #[token("else")] KW_ELSE,
    #[token("let")] KW_LET,

    #[token(";")] TOK_SEMI,
    #[token("(")] TOK_LPAREN,
    #[token(")")] TOK_RPAREN,
    #[token("{")] TOK_LBRACE,
    #[token("}")] TOK_RBRACE,
    #[token(",")] TOK_COMMA,

    #[token("-")] OP_MINUS,
    #[token("+")] OP_PLUS,
    #[token("*")] OP_STAR,
    #[token("/")] OP_SLASH,
    #[token("=")] OP_EQ,
    #[token("||")] OP_OR,
    #[token("&&")] OP_AND,
    #[token("==")] OP_EQEQ,
    #[token("!=")] OP_NEQ,
    #[token("<")] OP_LT,
    #[token("<=")] OP_LE,
    #[token(">")] OP_GT,
    #[token(">=")] OP_GE,
    #[token("!")] OP_BANG,

    #[regex(r"0|([1-9](0-9)*)")]
    LIT_INT,

    #[regex(r#""([^"\\]|\\.)*""#)]
    LIT_STR,

    #[regex(r"[a-zA-Z_][a-zA-Z_0-9]*")]
    LIT_IDENT,

    TOK_ERROR,
    TOK_EOF,
}
```

`Logos` generates a lexer for us using the annotations on this enum. We have tokens for keywords,
punctuation characters, operators, and "literals" identified by a [regular expression](https://en.wikipedia.org/wiki/Regular_expression).
We also have two special tokens:
- `TOK_ERROR`, produced when the lexer can't match a given part of the source code against _any_ token, and
- `TOK_EOF`, produced when the lexer reaches the end of the file.

If you're curious about what it expands to, [here it is in a gist](https://gist.github.com/jhwlr/628c82e22ed5d0ee668c079706f3e9de).

<details>
<summary>Aside: Storing data in TokenKind</summary>

Our `TokenKind` variants don't carry any data. It is possible to get `Logos` to parse integers for us, and store the resulting value
inside the `TokenKind` enum. We're not going to use that feature. What we want is for the lexer to only _recognize_ tokens, and to
continue even when encountering an error. We'll be much better positioned to report errors in the parser, so we'll process the token's
lexeme there, turning a `LIT_INT` into an `i64`, for example.

It's also a bit more complicated to match on tokens when they may carry values. In my experience, doing it this way leads to leaner,
faster code. 

</details>

If you're a Rust enjoyer, you may cringe at my choice of naming for the enum variants. There's a good reason for it,
which is that the enum type is not referenced directly in the parser. Instead, we have a glob import for all of its
variants into the parser's file:

```rust
use crate::token::TokenKind::*;
```

This lets us write slightly terser code, but we need the variant names to not conflict with anything else in the file.
I hope you'll forgive the weird naming convention!

Now we can "write" our lexer:

```rust
fn lex(src: &str) -> Vec<Token> {
    TokenKind::lexer(src)
        .spanned()
        .map(|item| match item {
            (Ok(kind), span) => Token {
                kind,
                span: span.into(),
            },
            (Err(()), span) => Token {
                kind: TOK_ERROR,
                span: span.into(),
            },
        })
        .chain([Token {
            kind: TOK_EOF,
            span: Span::empty(),
        }])
        .collect()
}
```

`Logos` is doing a **lot** of heavy lifting here. An equivalent lexer would be a few hundred lines of code,
and it wouldn't be nearly as fast. At this point, the only reason this function exists is to preprocess
what `Logos` outputs into a _slightly_ nicer form.

## Parser == grammar police

Before we talk about parsing, we still need to talk about _what_ we'll be parsing. Human languages have grammar,
and so do programming languages. Just like a school teacher, our parser will only accept what is _grammatically valid_[^7].
For anything else, it will output syntax errors.

We know what `simp` syntax looks like, but what does its grammar look like? To describe the grammar of human language,
we often use more human language to do it. But that won't work for computers.

To describe a programming language grammar, we'll use a notation called
[Extended Backus-Naur form](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form) (EBNF).
The _E_ in EBNF effectively means we can do whatever we want, because there is no single[^9] standard!

Here's what `simp`'s formal grammar looks like:

```ebnf
Program = { Stmt } ;

Stmt = StmtFn | StmtLet | StmtExpr ;

StmtFn = "fn" Identifier "(" [ ParamList ] ")" Block ;
ParamList = Identifier { "," Identifier } ;

StmtLet = "let" Identifier "=" Expr ";" ;

StmtExpr = Expr ";" ;

Expr = ExprIf | ExprOr ;

ExprIf = "if" Expr Block "else" Block ;

ExprOr = ExprAnd { "||" ExprAnd } ;
ExprAnd = ExprEq { "&&" ExprEq } ;
ExprEq = ExprOrd { ( "==" | "!=" ) ExprOrd } ;
ExprOrd = ExprAdd { ( "<" | "<=" | ">" | ">=" ) ExprAdd } ;
ExprAdd = ExprMul { ( "+" | "-" ) ExprMul } ;
ExprMul = ExprUnary { ( "*" | "/" | "%" ) ExprUnary } ;
ExprUnary = [ "-" | "!" ] ExprPostfix ;
ExprPostfix = ExprPrimary { ExprCall } ;
ExprCall = "(" [ ArgList ] ")" ;
ArgList = Expr { "," Expr } ;

ExprPrimary = INTEGER
        | STRING
        | IDENTIFIER
        | Block
        | "(" Expr ")"
        ;

Block = "{" { Stmt } [ Expr ] "}" ;
```

To break it down:
- `name = ... ;` is a _production rule_, also called a _nonterminal_.
- Anything in `"quotes"` is a _terminal_, present in the code verbatim.
- Anything in `[ brackets ]` is optional.
- Anything in `{ braces }` is repeated, but also optional.
- `left | right` means "left or right", but not both. It can be chained.

We have two main categories of syntax: statements and expressions.

> “ Where an expression's main job is to produce a _value_,
> a statement's job is to produce an _effect_ „
>
> &nbsp;
>
> [Bob Nystrom](https://github.com/munificent) in his book
> [Crafing Interpreters](https://craftinginterpreters.com/the-lox-language.html#statements)[^12]

Some rules are omitted for brevity, such as identifiers[^13] and strings[^14].
They tend to expand into quite a few recursive rules, which doesn't really help us here, so we treat them as _terminals_.

We also have a special case in the parser for a _trailing expression_ in a block of code, which is the
implicit return value[^15] of that block. That case isn't in the grammar, either.

And finally, we do handle trailing commas[^16], which is also a bit cumbersome to write out in EBNF. If you see

```
rule = nonterminal { "," nonterminal }
```

assume it should handle trailing commas _somehow_.

## A `simp`le parser

The purpose of a parser is:

> Given a valid sequence of tokens, produce an AST.

There are _many_ ways to do this. It's a [whole field of study](https://en.wikipedia.org/wiki/Parsing), one we're not going to
delve very deeply into right now.

The technique I've chosen for this parser is called [**recursive descent**](https://en.wikipedia.org/wiki/Recursive_descent_parser).
It's simple, very common, and very effective. Another common technique is called [pratt parsing](https://matklad.github.io/2020/04/13/simple-but-powerful-pratt-parsing.html).
Sometimes multiple different kinds of parsing techniques are combined together into a single super-parser.

Our parser will read the source code, one token at a time, and traverse the grammar. For each rule we traverse, we'll either
successfully match it, or produce an error:

<iframe
    src="/intro-to-parsing/step-by-step.html"
    style="border:none; width: 100%; height: 193px; overflow: hidden;"
    onload="this.style.height=(this.contentWindow.document.body.scrollHeight+40)+'px';"
></iframe>

In most of our parser, all we'll have to do is match on the current token,
and advance to the next position if we find the right one.

To help us with that, we'll need a cursor to keep track of which token is:

```rust
struct Cursor<'src> {
    code: &'src str,
    tokens: Vec<Token>,
    position: usize,
}
```

And a few methods on the cursor to work with our token list:

```rust
impl<'src> Cursor<'src> {
    /// Advance the cursor to the next token.
    fn advance(&mut self) {
        if self.position + 1 >= self.tokens.len() {
            return;
        }

        self.position += 1;
    }

    /// Returns the token under the cursor.
    fn current(&self) -> Token {
        self.tokens[self.position]
    }

    /// Returns the token before the cursor.
    fn previous(&self) -> Token {
        self.tokens[self.position - 1]
    }

    /// Returns the current token kind,
    /// shorthand for `c.current().kind`.
    fn kind(&self) -> TokenKind {
        self.current().kind
    }

    /// Returns `true` if the current token matches `kind`.
    fn at(&self, kind: TokenKind) -> bool {
        self.current().kind == kind
    }

    /// Returns `true` if the previous token matched `kind`.
    fn was(&self, kind: TokenKind) -> bool {
        self.position > 0 && self.previous().kind == kind
    }

    /// Returns `true` and advances
    /// if the current token matches `kind`.
    fn eat(&mut self, kind: TokenKind) -> bool {
        if self.at(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Returns the current token if it matches `kind`,
    /// otherwise returns an error.
    fn must(&mut self, kind: TokenKind) -> Result<Token> {
        let current = self.current();
        if self.eat(kind) {
            Ok(current)
        } else {
            error(
                format!(
                    "expected {kind:?}, found {:?}",
                    current.kind,
                ),
                current.span,
            )
            .into()
        }
    }
}
```

It's a sizeable chunk of code, but will save us a lot of typing soon.

## Syntax trees and recursive descent

The entrypoint for our parser will be a `parse` function. It turns a string into a program, or a syntax error.

```rust
fn parse(code: &str) -> Result<Program> {
    let cursor = &mut Cursor {
        code,
        tokens: lex(code),
        position: 0,
    };

    parse_program(c)
}
```

The process of hand-writing a recursive descent parser involves closely following the grammar, implementing it one
production rule at a time. We'll also define each node of the syntax tree on demand.

Our entrypoint calls into `parse_program`. Starting with its syntax tree node:

```rust 
struct Program {
    body: Vec<Stmt>,
    tail: Option<Expr>,
}
```

A program is a list of statements. We'll continue parsing statements until we reach the end of the file.

```rust
// Program = { Stmt } ;
fn parse_program(c: &mut Cursor) -> Result<Program> {
    let mut body = Vec::new();
    while !c.at(TOK_EOF) {
        body.push(parse_stmt(c)?);
    }

    // ...
}
```

<details>
<summary>Aside: Trailing expressions</summary>

We're trying to mimic Rust, which can be a bit tricky. A block of code can have a trailing expression.
That happens when it ends with an expression, and that expression isn't terminated by a semicolon:

```rust
{ { 1 } { 2 } } // valid syntax, implicit return value is `2`
```

Developing parsers (and compilers in general) often involves thinking about contrived examples like these.
Parsers tend to have a lot of [emergent behavior](https://en.wikipedia.org/wiki/Emergence).
It's your job to rein it in!

You can add parser complexity to allow users of your language to write
fewer characters. You can also keep things more verbose and explicit, and make your own life as a compiler
developer easier, because a verbose feature which solves a problem is still probably better than not solving
the feature at all. Sometimes there's a clever way to achieve both fewer characters and a simpler compiler.

In this case, we're adding parser complexity in exchange for the ability to write code like this:

```rust
{
    let a = 1;

    {
        // more tightly scoped thing
        let secret = 42;
        foo(a + secret)
    } // no semicolon, despite being an expression

    a // correctly parsed as the trailing expression for this block
}
```

Because we don't want to require block-like expressions be terminated by a semicolon, we need to do a few things:
- Any statement which _requires_ a semicolon, such as `let`, must consume it.
- Any non-trailing expression in statement position must require a semicolon.

We can determine if an expression in statement position is trailing by checking what comes after it:

- `EOF` - it's the end of the file, so it's trailing.
- `}` - it's the end of a block, also trailing.
- Anything else must be another statement or expression, so we require a semicolon to separate them.

This is a consequence of how `simp`'s grammar is laid out. It shows why writing a (mostly) formal grammar is useful!
You can reason about the entirety of your syntax, and resolve ambiguities.

{{ hr(data_content="end aside") }}

</details>

A statement is either a function, a variable, or an expression. To determine what we're looking at,
we need to match on the current token:

```rust
// Stmt = StmtFn | StmtLet | StmtExpr ;

enum Stmt {
    Fn(Box<StmtFn>),
    Let(Box<StmtLet>),
    Expr(Box<Expr>),
}

fn parse_stmt(c: &mut Cursor) -> Result<Stmt> {
    match c.kind() {
        KW_FN => parse_stmt_fn(c),
        KW_LET => parse_stmt_let(c),
        _ => parse_stmt_expr(c),
    }
}
```

A function has a name, a list of parameters, and a body.

```rust
struct StmtFn {
    name: String,
    params: Vec<String>,
    body: Block,
}

fn parse_stmt_fn(c: &mut Cursor) -> Result<Stmt> {
    assert!(c.eat(KW_FN));    
    let name = parse_ident(c)?;
    let params = parse_param_list(c)?;
    let body = parse_block(c)?;
    Ok(Stmt::Fn(Box::new(StmtFn { name, params, body })))
}
```

The assert may seem strange, but `parse_stmt` does not _consume_ the token it matches on.
The sub-parser it dispatches expects to already be in the right context, so we assert on it.

Identifiers are terminal symbols. To parse one, we must retrieve the lexeme of a `LIT_IDENT` token:

```rust
fn parse_ident(c: &mut Cursor) -> Result<String> {
    let token = c.must(LIT_IDENT)?;
    Ok(c.lexeme(token).to_owned())
}
```

Parameters are a parenthesized, comma-separated list of identifiers:

```rust
fn parse_param_list(c: &mut Cursor) -> Result<Vec<String>> {
    let mut list = Vec::new();
    c.must(TOK_LPAREN)?;
    // stop early if we have an empty list: `()`
    if !c.at(TOK_RPAREN) {
        loop {
            list.push(parse_ident(c)?);
            // stop if there's no comma,
            // or if the comma is trailing.
            if !c.eat(TOK_COMMA) || c.at(TOK_RPAREN) {
                break;
            }
        }
    }
    c.must(TOK_RPAREN)?;
    Ok(list)
}
```

It's a little more involved, because we also want to allow trailing commas.
We'll actually want to re-use that code later for argument lists, so let's break it out right now:

```rust
fn parse_paren_list<F, T>(
    c: &mut Cursor,
    mut elem: F,
) -> Result<Vec<T>>
where
    F: FnMut(&mut Cursor) -> Result<T>,
{
    let mut list = Vec::new();
    c.must(TOK_LPAREN)?;
    if !c.at(TOK_RPAREN) {
        loop {
            list.push(elem(c)?);
            if !c.eat(TOK_COMMA) || c.at(TOK_RPAREN) {
                break;
            }
        }
    }
    c.must(TOK_RPAREN)?;
    Ok(list)
}
```

And use that instead in `parse_param_list`:

```rust
fn parse_param_list(c: &mut Cursor) -> Result<Vec<String>> {
    parse_paren_list(c, parse_ident)
}
```

A function's body is a block. It has an identical definition to `Program`:

```rust
struct Block {
    body: Vec<Stmt>,
    tail: Option<Expr>, // see aside about trailing expressions
}
```

And its implementation is largely the same, except that now we also require braces to wrap the contents:

```rust
fn parse_block(c: &mut Cursor) -> Result<Vec<Block>> {
    c.must(TOK_LBRACE)?;
    let mut body = Vec::new();
    while !c.at(TOK_RBRACE) {
        body.push(parse_stmt(c)?);
    }
    c.must(TOK_RBRACE)?;

    // ...
}
```

Okay, I think you get the point. Recursively traverse the rules while consuming tokens.
Let's skip ahead to something a bit more interesting.

## Parsing expressions

To parse arithmetic, we have to think about [_associativity_](https://en.wikipedia.org/wiki/Operator_associativity)
and [_precedence_](https://en.wikipedia.org/wiki/Order_of_operations). 
In our case, both are already encoded within the grammar:

```
(* some parts omitted for brevity *)

ExprAdd = ExprMul { ( "+" | "-" ) ExprMul } ;
ExprMul = ExprUnary { ( "*" | "/" | "%" ) ExprUnary } ;
ExprUnary = [ "-" | "!" ] ExprPostfix ;
ExprPostfix = ExprPrimary { ExprCall } ;

ExprPrimary = 
        | IDENTIFIER
        ;
```

- If a rule A "contains" another rule B, then rule A has _lower_ precedence.
- In the absence of parentheses, operators with higher precedence are grouped
first.
- Associativity tells us how operators of the same precedence are grouped;
for left-associative operators like `+` and `*`, the left side takes precedence.

Consider an expression such as `a + b * d - c`:

<iframe
    src="/intro-to-parsing/step-by-step-2.html"
    style="border:none; width: 100%; height: 216px; overflow: hidden;"
    onload="this.style.height=(this.contentWindow.document.body.scrollHeight+40)+'px';"
></iframe>

```rust
enum Expr {
    Binary(Box<ExprBinary>),
    // ... more expressions
}

struct ExprBinary {
    lhs: Expr,
    op: BinaryOp,
    rhs: Expr,
}

enum BinaryOp {
    Add,
    Subtract,
    // ... other operators
}

fn parse_expr_add(c: &mut Cursor) -> Result<Expr> {
    let mut lhs = parse_expr_mul(c)?;
    loop {
        let op = match c.kind() {
            OP_PLUS => BinaryOp::Add,
            OP_MINUS => BinaryOp::Subtract,
            _ => break,
        };
        c.advance(); // eat `op`
        let rhs = parse_expr_mul(c)?;
        lhs = Expr::Binary(Box::new(ExprBinary { lhs, op, rhs }))
    }
    Ok(lhs)
}
```

We'll first recurse into `parse_expr_mul`, and then enter a loop.
The loop attempts to match one of the valid operators at this precedence
level, and converts them to their corresponding `BinaryOp` variants.

If we can't match anything, we exit the loop. The `break` happens _before_ we
consume the operator token, which means that if there is an expression with lower precedence
level which _can_ consume the operator, it will find it once we return to it.

```rust
fn parse_expr_mul(c: &mut Cursor) -> Result<Expr> {
    let mut lhs = parse_expr_unary(c)?;
    loop {
        let op = match c.kind() {
            OP_STAR => BinaryOp::Multiply,
            OP_SLASH => BinaryOp::Divide,
            _ => break,
        };
        c.advance(); // eat `op`
        let rhs = parse_expr_unary(c)?;
        lhs = Expr::Binary(Box::new(ExprBinary { lhs, op, rhs }))
    }
    Ok(lhs)
}
```

This looks very familiar, but matches different operators, and calls `parse_expr_unary`,
climbing higher in precedence.

```rust
fn parse_expr_unary(c: &mut Cursor) -> Result<Expr> {
    let op = match c.kind() {
        OP_MINUS => UnaryOp::Minus,
        OP_BANG => UnaryOp::Not,
        _ => return parse_expr_postfix(c),
    };
    let rhs = parse_expr_unary(c)?;
    Ok(Expr::Unary(Box::new(ExprUnary { op, rhs })))
}
```

For unary expressions like `-a` and `!a`, the operator comes first, and _then_ its `rhs` sub-expression,
where it loops back around to itself. In case we don't match any operators, we call `parse_expr_postfix`.

```rust
fn parse_expr_postfix(c: &mut Cursor) -> Result<Expr> {
    let mut expr = parse_expr_primary(c)?;
    while c.at(TOK_LPAREN) {
        expr = parse_expr_call(c, expr)?;
    }
    Ok(expr)
}
```

The parentheses come _after_ the callee, like `f(a, b, c)`.

```rust
fn parse_expr_call(c: &mut Cursor, callee: Expr) -> Result<Expr> {
    let args = parse_arg_list(c)?;
    Ok(Expr::Call(Box::new(ExprCall { callee, args })))
}

fn parse_arg_list(c: &mut Cursor) -> Result<Vec<Expr>> {
    let args = parse_paren_list(c, parse_expr)?;
    Ok(args)
}
```

There's `parse_paren_list` again, nice. But this time each element is an expression.
This is why the technique is called _recursive_ descent - we're recursing back into one of the rules
which we descended from.

When nesting expressions like `a(b(c()))`, each time we come across a call,
we'll try to parse an argument list, where we recurse into `parse_expr` for each argument.
That'll lead us back to another call expression. After parsing the innermost `c()`, we return
back to `b(...)`, which is now complete too, so back out to `a(...)` we go. This process repeats
no matter how many levels deep we are[^17].

```rust
fn parse_expr_primary(c: &mut Cursor) -> Result<Expr> {
    match c.kind() {
        LIT_INT => parse_expr_int(c),
        LIT_STR => parse_expr_str(c),
        LIT_IDENT => parse_expr_ident(c),
        TOK_LBRACE => parse_expr_block(c),
        TOK_LPAREN => parse_expr_group(c),
        _ => error(
            format!("unexpected token: {:?}", c.kind()),
            c.current().span,
        )
        .into(),
    }
}
```

There are more cases here than anywhere else. Let's look at two of them:

```rust
fn parse_expr_int(c: &mut Cursor) -> Result<Expr> {
    let token = c.must(LIT_INT)?;
    let value = c
        .lexeme(token)
        .parse()
        .map_err(|err| error(format!("failed to parse integer: {err}"), token.span))?;
    Ok(Expr::Int(Box::new(ExprInt { value })))
}

fn parse_expr_block(c: &mut Cursor) -> Result<Expr> {
    let inner = parse_block(c)?;
    Ok(Expr::Block(Box::new(ExprBlock { inner })))
}
```

Integers come from `LIT_INT` tokens. We retrieve the token's lexeme, which is the integer as a string,
and use the Rust standard library to parse it into a value. `parse_expr_str` and `parse_expr_ident`
work much the same way.

Block expressions are wrappers around blocks, re-using the same parsing code as the function body in `parse_stmt_fn`.
Blocks are lists of statements, which is another place where we recurse in our _recursive_ descent.

A few nice consequences stem from this:

- functions may be declared within other functions
- blocks can be nested within other blocks

...without us really having to explicitly handle that in any meaningful way[^11]. That's emergent behavior!

We could choose to rein it in, for example by not recursing into `parse_stmt`, but writing a `parse_stmt_except_fn`
which excludes it, and calling that instead. Then our parser would reject any functions declared within other functions.
Though we'd have to be careful to _never_ recurse into `parse_stmt` unless we absolutely mean to.

## Closing

That's as far as we'll go in this article. We have a `simp`le parser, which is hopefully easy to understand.
We went a bit beyond a _truly_ simple grammar, like [Lisp](https://en.wikipedia.org/wiki/Lisp_(programming_language))
or [ML](https://en.wikipedia.org/wiki/ML_(programming_language)). 

Our parser is probably not very efficient! I haven't benchmarked it at all, but we're doing many small allocations
all over the place. That can't be good. I'd like to follow this article up with one about a _flatter_ memory
representation for `simp`'s AST, and some benchmarks with colorful graphs.

I'll leave you with a little REPL you can use to test out the parser:

**TODO**: REPL

The code is [available on GitHub](https://github.com/jhwlr/simp).

{{ hr(data_content="footnotes")}}

[^1]: Before I get flamed: This is a _gross_ oversimplification.
[^10]: Or at least how I understand it, I'm no academic!
[^2]: You can also parse straight into executable code.
[^3]: Concrete syntax trees are often used in [language servers](https://en.wikipedia.org/wiki/Language_Server_Protocol).
[^4]: Any similarity in name to existing programming languages is purely coincidental, I looked it up and didn't find anything.
[^5]: We're going all in on expressions with no `return` keyword! This complicates the parser a bit, but also makes it more
interesting.
[^6]: Our `Span` type is like `Range<u32>`, but it implements `Copy`. The standard library can't break the `Range` type, so
we're stuck with making our own. It doesn't have first-class syntax, but at least it can still be used to directly index into strings.
[^8]: We could also read one Rust `char` at a time. We won't allow unicode outside of strings, and spans are easier to produce when dealing
with bytes, so we're not doing that.
[^7]: It's possible to write a parser that doesn't stop at anything, and treats errors as part of the syntax tree. Another technique common in language servers!
[^9]: [Attempts](https://www.iso.org/standard/26153.html) [have been](https://www.cl.cam.ac.uk/~mgk25/iso-ebnf.html) [made](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form) to standardise it,
but nothing seems to have stuck. Different tools accept different syntaxes, and research papers are similarly inconsistent.
[^12]: I cannot recommend this book enough. The web version was my entrypoint into programming languages.
[^13]: An identifier is a name given to some construct, like a function or variable.
[^14]: A string is a sequence of characters, in our case wrapped in double quotes. They're used to store arbitrary text directly in source code. 
[^15]: Meaning there is no `return` keyword which tells the compiler that you want a specific value to be the return value.
[^16]: These may appear in a few places, such as function calls (`f(a, b, c,)`) and parameter lists (`fn f(a, b, c,) {}`).
[^17]: Until we hit a [stack overflow](https://en.wikipedia.org/wiki/Stack_overflow). There are ways to solve that. We could [manually allocate our own stack](https://docs.rs/stacker/latest/stacker/)
before entering the parser, and grow it when we are about to run out of space.
[^11]: Well, at least in the parser. If functions can be declared in any scope, that means you have to track them in every scope
in subsequent compilation stages. 
