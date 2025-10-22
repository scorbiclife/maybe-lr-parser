from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import cache, cached_property
from typing import (
    Iterable,
    Collection,
    Generator,
    Literal,
    Sequence,
    TypeIs,
)


type End = Literal["$"]
type Accept = Literal["Accept"]
type Start = Literal["^"]

type TerminalSymbol = str | Accept | Start | End
type NonTerminalSymbol = str

type BaseSymbol = TerminalSymbol | NonTerminalSymbol
type DerivedSymbol = tuple[BaseSymbol, BaseSymbol, *tuple[BaseSymbol, ...]]
type Symbol = BaseSymbol | DerivedSymbol


class SymbolUtils:
    @staticmethod
    def is_base_symbol(s: Symbol) -> TypeIs[BaseSymbol]:
        return not SymbolUtils.is_derived_symbol(s)

    @staticmethod
    def is_derived_symbol(s: Symbol) -> TypeIs[DerivedSymbol]:
        return isinstance(s, tuple)


def dfs(node, get_next_nodes, visited=None):
    if visited is None:
        visited = set()

    if node in visited:
        return

    visited.add(node)
    yield node

    for next_node in get_next_nodes(node):
        yield from dfs(next_node, get_next_nodes, visited)


type Production = UnaryProduction | BinaryProduction | TernaryOrMoreProduction


class ProductionUtils:
    @staticmethod
    def from_lhs_and_rhs(lhs, rhs):
        if len(rhs) == 0:
            raise ValueError("This parser does not accept empty productions.")
        elif len(rhs) == 1:
            return UnaryProduction(lhs, rhs[0])
        elif len(rhs) == 2:
            return BinaryProduction(lhs, rhs[0], rhs[1])
        else:
            return TernaryOrMoreProduction(lhs, rhs[0], tuple(rhs[1:]))

    @staticmethod
    def from_derived_symbol(s: DerivedSymbol):
        # Use the tuple itself as the symbol for the lhs
        try:
            a, b, c, *rest = s  # type: ignore
            return TernaryOrMoreProduction(s, a, (b, c, *rest))
        except ValueError:
            pass
        try:
            a, b = s  # type: ignore
            return BinaryProduction(s, a, b)
        except ValueError:
            raise ValueError("Expected derived symbol, got something else")

    @staticmethod
    def from_three_symbols(lhs: Symbol, rhs0: BaseSymbol, rhs1: Symbol):
        if SymbolUtils.is_base_symbol(rhs1):
            return BinaryProduction(lhs, rhs0, rhs1)
        else:
            return TernaryOrMoreProduction(lhs, rhs0, rhs1)


@dataclass(frozen=True)
class UnaryProduction:
    lhs: Symbol
    rhs0: BaseSymbol

    def __str__(self):
        return f"Production({self.lhs} -> {self.rhs0})"


@dataclass(frozen=True)
class BinaryProduction:
    lhs: Symbol
    rhs0: BaseSymbol
    rhs1: BaseSymbol

    def __str__(self):
        return f"Production({self.lhs} -> {self.rhs0} {self.rhs1})"


@dataclass(frozen=True)
class TernaryOrMoreProduction:
    """Separated from `BinaryProduction` because it has derived productions"""

    lhs: Symbol
    rhs0: BaseSymbol
    rhs1: DerivedSymbol

    def __str__(self):
        return f"Production({self.lhs} -> {self.rhs0} {self.rhs1})"


type State = BaseState | DerivedState


class StateUtils:
    @staticmethod
    def createState(s: Symbol, l: TerminalSymbol) -> State:
        return (
            DerivedState(s, l) if SymbolUtils.is_derived_symbol(s) else BaseState(s, l)
        )


@dataclass(frozen=True)
class BaseState:
    symbol: BaseSymbol
    follow: TerminalSymbol

    def __str__(self):
        return f"{self.symbol}[{self.follow}]"


@dataclass(frozen=True)
class DerivedState:
    symbol: DerivedSymbol
    follow: TerminalSymbol

    def __str__(self):
        return f"{self.symbol}[{self.follow}]"


type Reduction = UnaryReduction | BinaryReduction


class ReductionUtils:
    pass


@dataclass(frozen=True)
class UnaryReduction:
    source: BaseState
    destination: State

    def __str__(self):
        return f"Reduction({self.source} -> {self.destination})"

    def production(self) -> Production:
        lhs = self.destination.symbol
        rhs0 = self.source.symbol
        return UnaryProduction(lhs, rhs0)


@dataclass(frozen=True)
class BinaryReduction:
    source: BaseState
    destination: State
    right_dependency: State

    def __str__(self):
        return f"Reduction({self.source} {self.right_dependency} -> {self.destination})"

    def production(self) -> Production:
        lhs = self.destination.symbol
        rhs0 = self.source.symbol
        rhs1 = self.right_dependency.symbol
        return ProductionUtils.from_three_symbols(lhs, rhs0, rhs1)


ACCEPT: Accept = "Accept"
START: Start = "^"
END: End = "$"


@dataclass(frozen=True)
class Grammar:
    nonterminal_symbols: frozenset[NonTerminalSymbol]
    terminal_symbols: frozenset[TerminalSymbol]
    productions: tuple[Production, ...]
    start_symbol: NonTerminalSymbol

    @cached_property
    def start_production(self):
        return ProductionUtils.from_lhs_and_rhs(ACCEPT, [START, self.start_symbol])

    @cached_property
    def basic_symbols(self) -> frozenset[BaseSymbol]:
        return self.terminal_symbols | self.nonterminal_symbols

    @cached_property
    def special_symbols(self) -> frozenset[BaseSymbol]:
        return frozenset([START, ACCEPT, END])

    @cached_property
    def all_symbols(self) -> frozenset[BaseSymbol]:
        return self.basic_symbols | self.special_symbols

    @cache
    def first(self, symbol: BaseSymbol) -> frozenset[TerminalSymbol]:
        return frozenset(
            s
            for s in dfs(symbol, self._direct_first_symbol)
            if s in self.terminal_symbols
        )

    @cache
    def _direct_first_symbol(self, symbol: BaseSymbol) -> frozenset[BaseSymbol]:
        return frozenset(p.rhs0 for p in self.productions if p.lhs == symbol)


class ReductionGraph:
    """The graph of `State`s and `Reduction`s that appears from the given grammar"""

    def __init__(self, grammar: Grammar):
        self.grammar = grammar
        self.states, self.reductions = self._init_states_and_reductions()

    def possible_reductions_for(
        self, start: State, end: State
    ) -> Collection[Reduction]:
        return set(
            r
            for r in self.direct_reductions_from(start)
            if self.can_be_eventually_reduced_to(r.destination, end)
        )

    @cache
    def can_be_eventually_reduced_to(self, start: State, end: State) -> bool:
        return any(s == end for s in dfs(start, self.direct_reduced_states_from))

    def direct_reduced_states_from(self, start: State):
        return frozenset(r.destination for r in self.direct_reductions_from(start))

    @cache
    def direct_reductions_from(self, start: State) -> frozenset[Reduction]:
        return frozenset(r for r in self.reductions if r.source == start)

    @cache
    def first(self, s: Symbol) -> frozenset[TerminalSymbol]:
        return self.grammar.first(s[0] if SymbolUtils.is_derived_symbol(s) else s)

    # Lower-level helper methods

    def _init_states_and_reductions(
        self,
    ) -> tuple[frozenset[State], frozenset[Reduction]]:
        known_states = set()
        reductions: list[Reduction] = []
        stack: list[State] = [BaseState(ACCEPT, END)]

        while stack:
            state = stack.pop()
            if state in known_states:
                continue
            known_states.add(state)
            reductions.extend(self._reductions_derived_from_lhs(state))
            stack.extend(self._all_states_derived_from_lhs(state))

        return (frozenset(known_states), frozenset(reductions))

    def _all_states_derived_from_lhs(self, state: State) -> frozenset[State]:
        result: list[State] = []
        for r in self._reductions_derived_from_lhs(state):
            result.append(r.source)
            if isinstance(r, BinaryReduction):
                result.append(r.right_dependency)
        return frozenset(result)

    @cache
    def _reductions_derived_from_lhs(self, dest: State) -> frozenset[Reduction]:
        return frozenset(
            r
            for p in self._productions_with_lhs(dest.symbol)
            for r in self._all_reductions_derived_from(p, dest.follow)
        )

    def _productions_with_lhs(self, symbol: Symbol) -> Iterable[Production]:
        if SymbolUtils.is_derived_symbol(symbol):
            return (ProductionUtils.from_derived_symbol(symbol),)

        if symbol not in self.grammar.all_symbols:
            raise ValueError(f"Unknown symbol: {symbol}")

        return (
            (self.grammar.start_production,)
            if symbol == ACCEPT
            else tuple(p for p in self.grammar.productions if p.lhs == symbol)
        )

    def _all_reductions_derived_from(
        self, p: Production, lookahead: TerminalSymbol
    ) -> Generator[Reduction]:
        yield from self._direct_reduction_of(p, lookahead)
        if isinstance(p, UnaryProduction):
            return
        if SymbolUtils.is_base_symbol(p.rhs1):
            return
        indirect_reductions = self._all_reductions_derived_from(
            ProductionUtils.from_derived_symbol(p.rhs1), lookahead
        )
        yield from indirect_reductions

    def _direct_reduction_of(
        self, p: Production, lookahead: TerminalSymbol
    ) -> Generator[Reduction]:
        # Production(A -> B)[x] => Reduction(B[x] -> A[x])
        if isinstance(p, UnaryProduction):
            dest = StateUtils.createState(p.lhs, lookahead)
            source = BaseState(p.rhs0, lookahead)
            yield UnaryReduction(source, dest)
            return
        # Production(A -> B C)[x] => Reduction(B[first(C)] C[x] -> A[x])
        for f in self.first(p.rhs1):
            dest = StateUtils.createState(p.lhs, lookahead)
            source = BaseState(p.rhs0, follow=f)
            right_dependency = StateUtils.createState(p.rhs1, lookahead)
            yield BinaryReduction(source, dest, right_dependency)


type ParseAction = Shift | Reduce


class ParseActionUtils:
    class Conflict(BaseException):
        actions: Collection[ParseAction]

    class Failure(BaseException):
        pass

    @staticmethod
    def action_for_reduction(r: Reduction):
        return Reduce(r) if isinstance(r, UnaryReduction) else Shift()

    @staticmethod
    def only_action_of(actions: Collection[ParseAction]) -> ParseAction:
        action_set = set(actions)
        if not action_set:
            raise ParseActionUtils.Failure()
        if len(action_set) > 1:
            raise ParseActionUtils.Conflict(action_set)
        return action_set.pop()


@dataclass(frozen=True)
class Shift:

    def __str__(self):
        return "Shift()"


@dataclass(frozen=True)
class Reduce:
    reduction: Reduction

    def __str__(self):
        return f"Reduce({self.reduction})"


@dataclass
class ParseContext:
    """Context for determining next actions. This is an item for our parse stack."""

    current_state: State
    goals: dict[State, Reduction]

    @property
    def mandatory_reduction(self) -> Reduction | None:
        """If there is a goal that is achieved currently, the corresponding reduction is mandatory"""
        return self.goals.get(self.current_state, None)

    def create_next_context(
        self,
        reductions: Collection[Reduction],
        lookahead: TerminalSymbol,
    ) -> "ParseContext":
        next_context = ParseContext(
            BaseState(self.current_state.follow, lookahead), dict()
        )
        for r in reductions:
            assert isinstance(r, BinaryReduction)
            assert r.source == self.current_state
            next_goal_state = r.right_dependency
            if next_context.goals.get(next_goal_state, r) != r:
                raise ParseActionUtils.Conflict(
                    [next_context.goals[next_goal_state], r]
                )
            next_context.goals[next_goal_state] = r
        return next_context


class Source(ABC):
    @abstractmethod
    def lookahead(self):
        raise NotImplementedError()

    @abstractmethod
    def shift(self):
        raise NotImplementedError()


class SymbolSequenceSource(Source):
    def __init__(self, source: Sequence[Symbol]):
        self.source = source
        self.index = 0

    def __getitem__(self, index: int):
        return self.source[index] if index < len(self.source) else END

    def lookahead(self):
        return self[self.index]

    def shift(self):
        self.index += 1


class Parser:
    GOAL_STATE = BaseState(ACCEPT, END)

    def __init__(
        self, grammar: Grammar, source: Source, graph: ReductionGraph | None = None
    ):
        self.grammar = grammar
        self.graph = graph or ReductionGraph(grammar)
        self.source = source
        initial_state = BaseState(START, self.source.lookahead())
        initial_context = ParseContext(
            initial_state, {Parser.GOAL_STATE: None}  # type: ignore
        )
        assert initial_context.mandatory_reduction is None
        self.stack = [initial_context]

    def parse(self) -> Generator[ParseAction]:
        while not self.success():
            yield self.step()

    def success(self):
        return (
            len(self.stack) == 1
            and self.current_context.current_state == Parser.GOAL_STATE
        )

    def step(self) -> ParseAction:
        """Not all reductions of the derived grammar are reductions of the original grammar"""
        possible_reductions = self.possible_reductions()
        actions: set[ParseAction] = set(
            ParseActionUtils.action_for_reduction(r) for r in possible_reductions
        )
        mandatory_reduction = self.current_context.mandatory_reduction
        if mandatory_reduction is not None:
            actions.add(Reduce(mandatory_reduction))

        action = ParseActionUtils.only_action_of(actions)

        if isinstance(action, Shift):
            return self.shift(next_goals=possible_reductions)
        elif isinstance(action.reduction, UnaryReduction):
            return self.reduce1(action.reduction)
        else:
            return self.reduce2(action.reduction)

    def shift(self, next_goals: Collection[Reduction]) -> ParseAction:
        self.source.shift()
        next_context = self.current_context.create_next_context(
            next_goals, self.source.lookahead()
        )
        self.stack.append(next_context)
        return Shift()

    @property
    def current_context(self) -> ParseContext:
        return self.stack[-1]

    def reduce1(self, r: UnaryReduction):
        assert self.current_context.current_state == r.source
        self.current_context.current_state = r.destination
        return Reduce(r)

    def reduce2(self, r: BinaryReduction):
        assert self.current_context.current_state == r.right_dependency
        self.stack.pop()
        self.current_context.current_state = r.destination
        return Reduce(r)

    def possible_reductions(self) -> set[Reduction]:
        context = self.current_context
        return set(
            r
            for goal in context.goals
            for r in self.graph.possible_reductions_for(context.current_state, goal)
        )
