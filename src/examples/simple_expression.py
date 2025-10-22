from ..parser import *
from pprint import pprint as pp

def main():
    plus, times, identifier, literal = "+", "*", "id", "lit"
    terminal = frozenset([plus, times, identifier, literal])

    expression, term, value = "Expr", "Term", "Value"
    nonterminal = frozenset([expression, term, value])

    productions: tuple[Production, ...] = tuple(
        ProductionUtils.from_lhs_and_rhs(l, r)
        for l, r in [
            (expression, [term]),
            (expression, [term, plus, expression]),
            (term, [value]),
            (term, [term, times, value]),
            (value, [identifier]),
            (value, [literal]),
        ]
    )

    start_symbol = expression

    grammar = Grammar(nonterminal, terminal, productions, start_symbol)

    graph = ReductionGraph(grammar)

    # example: A * 2 + 1
    source = SymbolSequenceSource([identifier, times, literal, plus, literal])

    parser = Parser(grammar, source)
    for action in parser.parse():
        print(action)

if __name__ == "__main__":
    main()