# Path: unplow/inbox/gemweb101.md
# Scoped Systems-Subset EBNF Grammars for C & Rust Parameter Extraction
# Max Column: 80 Columns

For logit-masking engines like Lark, Outlines, or custom LR(1)/Earley pushdown
automata, loading an unconstrained full language specification (e.g., standard
ISO C11 or rustc grammar) leads to state-space explosion and parser thrashing
during autoregressive decoding.

To harvest clean systems routines for `unplow`, the optimal approach is a
**scoped systems-subset EBNF**. These grammars focus directly on top-level
declarations, functions, memory manipulation, and structs, while enforcing
strict termination so routines close naturally before hitting token limits.

---

### 1. C Systems Subset EBNF (`c_systems.ebnf`)

This grammar covers C99/C11 low-level primitives: fixed-width types
(`uint32_t`, `size_t`), struct definitions, pointer manipulations, loops, and
bitwise logic.

```ebnf
?start: translation_unit

translation_unit: (include_directive | type_definition | function_definition)+

include_directive: "#include" ("<" IDENTIFIER ".h>" | "\"" IDENTIFIER ".h\"")

type_definition: "typedef" "struct" "{" (struct_member ";")+ "}" IDENTIFIER ";"
struct_member: type IDENTIFIER

function_definition: type IDENTIFIER "(" parameter_list? ")" compound_statement

parameter_list: parameter ("," parameter)*
parameter: ("const" WS)? type pointer_qualifier? IDENTIFIER
pointer_qualifier: "*" ("const")?

type: "void" | "int" | "char" | "size_t" | "uint8_t" | "uint16_t" 
    | "uint32_t" | "uint64_t" | "int8_t" | "int16_t" | "int32_t" | "int64_t"
    | "bool" | IDENTIFIER

compound_statement: "{" statement* "}"

statement: compound_statement
         | declaration_statement
         | assignment_statement
         | if_statement
         | for_statement
         | while_statement
         | return_statement

declaration_statement: type pointer_qualifier? IDENTIFIER ("=" expression)? ";"
assignment_statement: lvalue ("=" | "+=" | "-=" | "*=" | "&=" | "|=" | "^=" | "<<=" | ">>=") expression ";"

if_statement: "if" "(" expression ")" statement ("else" statement)?
for_statement: "for" "(" (declaration_statement | assignment_statement | ";") expression? ";" assignment_expr? ")" statement
while_statement: "while" "(" expression ")" statement
return_statement: "return" expression? ";"

lvalue: pointer_deref | IDENTIFIER ("[" expression "]")? ("." IDENTIFIER | "->" IDENTIFIER)*
pointer_deref: "*" (lvalue | "(" expression ")")

expression: conditional_expr
conditional_expr: logical_or ("?" expression ":" conditional_expr)?
logical_or: logical_and ("||" logical_and)*
logical_and: bitwise_or ("&&" bitwise_or)*
bitwise_or: bitwise_xor ("|" bitwise_xor)*
bitwise_xor: bitwise_and ("^" bitwise_and)*
bitwise_and: equality ("&" equality)*
equality: relational (("==" | "!=") relational)*
relational: shift (("<" | "<=" | ">" | ">=") shift)*
shift: additive (("<<" | ">>") additive)*
additive: multiplicative (("+" | "-") multiplicative)*
multiplicative: unary (("*" | "/" | "%") unary)*

unary: ("!" | "~" | "-" | "+" | "*" | "&" | "sizeof") unary
     | primary

primary: "(" expression ")"
       | function_call
       | lvalue
       | NUMBER
       | STRING_LITERAL

function_call: IDENTIFIER "(" (expression ("," expression)*)? ")"
assignment_expr: lvalue ("=" | "+=" | "-=" | "&=" | "|=") expression

IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
NUMBER: /0x[0-9a-fA-F]+|[0-9]+/
STRING_LITERAL: /"([^"\\]|\\.)*"/
WS: /[ \t\n\r]+/
%ignore WS

```

---

### 2. Rust Systems Subset EBNF (`rust_systems.ebnf`)

This grammar models memory-safe and systems routines: module imports (`use`),
`pub fn` and `fn`, explicit lifetimes, borrowing (`&mut`), struct definitions,
trait implementations (`impl Drop`, `impl Foo`), and deterministic pattern
matching.

```ebnf
?start: source_file

source_file: (use_declaration | struct_definition | impl_block | function_definition)+

use_declaration: "use" path ("::" "{" ident_list "}")? ";"
path: IDENTIFIER ("::" IDENTIFIER)*
ident_list: IDENTIFIER ("," IDENTIFIER)*

struct_definition: ("pub" WS)? "struct" IDENTIFIER generic_params? "{" (struct_field ",")* "}"
struct_field: ("pub" WS)? IDENTIFIER ":" type
generic_params: "<" generic_param ("," generic_param)* ">"
generic_param: LIFETIME | IDENTIFIER

impl_block: "impl" generic_params? (IDENTIFIER "for")? IDENTIFIER generic_params? "{" function_definition* "}"

function_definition: ("pub" WS)? ("unsafe" WS)? "fn" IDENTIFIER generic_params? "(" param_list? ")" ("->" type)? block

param_list: self_param ("," param)* | param ("," param)*
self_param: ("&" LIFETIME?)? ("mut" WS)? "self"
param: ("mut" WS)? IDENTIFIER ":" type

type: reference_type 
    | slice_type 
    | array_type 
    | tuple_type 
    | primitive_type 
    | path_type

primitive_type: "u8" | "u16" | "u32" | "u64" | "usize" 
              | "i8" | "i16" | "i32" | "i64" | "isize" 
              | "bool" | "f32" | "f64"

reference_type: "&" LIFETIME? ("mut" WS)? type
slice_type: "[" type "]"
array_type: "[" type ";" NUMBER "]"
tuple_type: "(" (type ("," type)*)? ")"
path_type: path ("<" type ("," type)* ">")?

block: "{" statement* expression? "}"

statement: let_statement
         | assignment_statement
         | expression_statement
         | return_statement

let_statement: "let" ("mut" WS)? IDENTIFIER (":" type)? ("=" expression)? ";"
assignment_statement: lvalue ("=" | "+=" | "-=" | "*=" | "&=" | "|=" | "^=" | "<<=" | ">>=") expression ";"
expression_statement: expression ";"
return_statement: "return" expression? ";"

expression: if_expression
          | match_expression
          | loop_expression
          | binary_expression
          | primary_expr

if_expression: "if" expression block ("else" (block | if_expression))?
match_expression: "match" expression "{" (match_arm ",")* "}"
match_arm: pattern "=>" (block | expression)
pattern: "_" | IDENTIFIER | path ("(" pattern_list? ")")? | NUMBER
pattern_list: pattern ("," pattern)*

loop_expression: ("while" expression | "loop") block

binary_expression: expression BIN_OP expression
BIN_OP: "||" | "&&" | "==" | "!=" | "<" | "<=" | ">" | ">="
      | "|" | "^" | "&" | "<<" | ">>" | "+" | "-" | "*" | "/" | "%"

primary_expr: "(" expression ")"
            | path ("(" (expression ("," expression)*)? ")")?
            | path "{" (field_init ",")* "}"
            | lvalue
            | NUMBER
            | STRING_LITERAL
            | "true" | "false"

field_init: IDENTIFIER ":" expression
lvalue: ("*" | "&" ("mut" WS)?)? IDENTIFIER ("." IDENTIFIER | "[" expression "]")*

IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
LIFETIME: /'[a-zA-Z_][a-zA-Z0-9_]*/
NUMBER: /0x[0-9a-fA-F]+|[0-9]+([uif](8|16|32|64|size))?/
STRING_LITERAL: /"([^"\\]|\\.)*"/
WS: /[ \t\n\r]+/
%ignore WS

```

---

### 3. Loading the Rules in `unplow`

In `paradigm1.py`, plug these specifications into your token-masking setup:

```python
from outlines.fsm.guide import CFGGuide

class DomainGrammarEngine:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.grammar_files = {
            "c": "unplow/grammars/c_systems.ebnf",
            "rust": "unplow/grammars/rust_systems.ebnf",
        }

    def build_guide(self, domain: str, initial_code: str) -> CFGGuide:
        with open(self.grammar_files[domain], "r", encoding="utf-8") as f:
            grammar_str = f.read()
            
        # The parser initializes its state and immediately shifts 
        # across the tokens in initial_code (e.g., your domain seed prefix),
        # guaranteeing that subsequent predictions strictly complete the AST.
        guide = CFGGuide(grammar_str, self.tokenizer)
        return guide

```

By constraining generation with these subsets, token sampling stays locked
inside legitimate C structs/routines and Rust type/lifetime parameters,
preventing syntax breakage while preserving the model's low-level systems
knowledge.

<!-- end of file: unplow/inbox/gemweb101.md -->

