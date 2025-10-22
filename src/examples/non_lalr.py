from ..parser import *
from pprint import pprint as pp

def main():
    terminal = frozenset("abcde")
    nonterminal = frozenset("SEF")

    # Source: https://en.wikipedia.org/wiki/LALR_parser
    productions: tuple[Production, ...] = tuple(
        ProductionUtils.from_lhs_and_rhs(l, r)
        for l, r in [
            ("S", ["a", "E", "c"]),
            ("S", ["a", "F", "d"]),
            ("S", ["b", "E", "d"]),
            ("S", ["b", "F", "c"]),
            ("E", ["e"]),
            ("F", ["e"]),
        ]
    )

    start_symbol = "S"

    grammar = Grammar(nonterminal, terminal, productions, start_symbol)

    source = SymbolSequenceSource("bed")

    parser = Parser(grammar, source)
    for action in parser.parse():
        print(action)

if __name__ == "__main__":
    main()