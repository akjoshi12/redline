from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

# Fitness = number of passing tests (higher is better)
FitnessType = int

MutationHook = Callable[[str], str]  # takes candidate code, returns mutated code


@dataclass(frozen=True)
class Candidate:
    task_id: str
    code: str
    fitness: FitnessType
    generation: int = 0
    index: int = 0


@dataclass
class Population:
    """Per-task candidate pool with tournament selection and mutation hooks."""

    task_id: str
    max_size: int = 8
    rng: random.Random = field(default_factory=random.Random)
    _candidates: List[Candidate] = field(default_factory=list, init=False)
    _next_index: int = field(default=0, init=False)

    @property
    def candidates(self) -> List[Candidate]:
        return list(self._candidates)

    @property
    def size(self) -> int:
        return len(self._candidates)

    @property
    def best_fitness(self) -> Optional[FitnessType]:
        if not self._candidates:
            return None
        return max(c.fitness for c in self._candidates)

    @property
    def is_solved(self) -> bool:
        """True if any candidate has fitness > 0 (at least one test passing)."""
        return self.best_fitness is not None and self.best_fitness > 0

    def add(
        self,
        code: str,
        fitness: FitnessType,
        generation: int = 0,
    ) -> Candidate:
        """Add a candidate. If at max_size, evict the worst before adding."""
        if len(self._candidates) >= self.max_size:
            self._evict_worst()

        idx = self._next_index
        self._next_index += 1

        candidate = Candidate(
            task_id=self.task_id,
            code=code,
            fitness=fitness,
            generation=generation,
            index=idx,
        )
        self._candidates.append(candidate)
        return candidate

    def select(self, tournament_size: int = 3) -> Optional[Candidate]:
        """Tournament selection: pick *tournament_size* random candidates, return the best."""
        if not self._candidates:
            return None

        k = min(tournament_size, len(self._candidates))
        pool = self.rng.sample(self._candidates, k)
        return max(pool, key=lambda c: c.fitness)

    def select_best(self) -> Optional[Candidate]:
        """Return the single best candidate (deterministic)."""
        if not self._candidates:
            return None
        return max(self._candidates, key=lambda c: c.fitness)

    def mutate(
        self,
        hook: MutationHook,
        generation: int = 0,
    ) -> Optional[Candidate]:
        """Select a candidate via tournament, apply mutation hook, add result."""
        parent = self.select()
        if parent is None:
            return None

        mutated_code = hook(parent.code)
        # Fitness unknown until evaluated; store as -1 sentinel.
        return self.add(mutated_code, fitness=-1, generation=generation)

    def mutate_best(
        self,
        hook: MutationHook,
        generation: int = 0,
    ) -> Optional[Candidate]:
        """Select the best candidate, apply mutation hook, add result."""
        parent = self.select_best()
        if parent is None:
            return None

        mutated_code = hook(parent.code)
        return self.add(mutated_code, fitness=-1, generation=generation)

    def _evict_worst(self) -> None:
        """Remove the candidate with lowest fitness."""
        worst_idx = min(
            range(len(self._candidates)), key=lambda i: self._candidates[i].fitness
        )
        self._candidates.pop(worst_idx)


@dataclass
class SwarmPopulation:
    """Manages populations across all tasks in a swarm run."""

    max_size: int = 8
    seed: int = 42
    _populations: Dict[str, Population] = field(default_factory=dict, init=False)

    def get_or_create(self, task_id: str) -> Population:
        if task_id not in self._populations:
            self._populations[task_id] = Population(
                task_id=task_id,
                max_size=self.max_size,
                rng=random.Random(self.seed),
            )
        return self._populations[task_id]

    def population_for(self, task_id: str) -> Optional[Population]:
        return self._populations.get(task_id)

    @property
    def solved_tasks(self) -> List[str]:
        return [tid for tid, pop in self._populations.items() if pop.is_solved]

    @property
    def unsolved_tasks(self) -> List[str]:
        return [tid for tid, pop in self._populations.items() if not pop.is_solved]
