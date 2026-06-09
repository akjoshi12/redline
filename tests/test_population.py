from __future__ import annotations

import random

from redline.swarm.population import (
    Candidate,
    FitnessType,
    MutationHook,
    Population,
    SwarmPopulation,
)


# --- Candidate ---


class TestCandidate:
    def test_fields(self):
        c = Candidate(task_id="t1", code="x=1", fitness=3, generation=2, index=5)
        assert c.task_id == "t1"
        assert c.code == "x=1"
        assert c.fitness == 3
        assert c.generation == 2
        assert c.index == 5

    def test_frozen(self):
        import pytest as pt

        c = Candidate(task_id="t1", code="x=1", fitness=0)
        with pt.raises(Exception):
            c.fitness = 99  # type: ignore


# --- Population.add ---


class TestPopulationAdd:
    def test_add_single(self):
        pop = Population(task_id="t1")
        c = pop.add("code_a", fitness=5)
        assert len(pop.candidates) == 1
        assert c.code == "code_a"
        assert c.fitness == 5

    def test_add_multiple(self):
        pop = Population(task_id="t1")
        pop.add("a", fitness=1)
        pop.add("b", fitness=3)
        pop.add("c", fitness=2)
        assert len(pop.candidates) == 3

    def test_evicts_worst_at_max(self):
        pop = Population(task_id="t1", max_size=3)
        pop.add("a", fitness=5)
        pop.add("b", fitness=1)
        pop.add("c", fitness=3)
        # Adding a 4th should evict the worst (fitness=1, "b")
        pop.add("d", fitness=2)
        assert len(pop.candidates) == 3
        codes = [c.code for c in pop.candidates]
        assert "b" not in codes

    def test_evicts_lowest_fitness(self):
        pop = Population(task_id="t1", max_size=2)
        pop.add("a", fitness=10)
        pop.add("b", fitness=7)
        # Adding with fitness 5 should evict the worst (fitness=5 is new, but we add first then check)
        # Actually: at size 2, adding triggers eviction of worst BEFORE adding.
        # So current worst = "a" (10), "b" (7). Worst is... wait no, min fitness.
        # Current: a=10, b=7. Adding d=5. Evict worst from existing: b(7) < a(10), evict b.
        # Then add d. Result: a(10), d(5).
        pop.add("d", fitness=5)
        assert len(pop.candidates) == 2
        codes = [c.code for c in pop.candidates]
        assert "a" in codes
        assert "d" in codes

    def test_best_fitness(self):
        pop = Population(task_id="t1")
        assert pop.best_fitness is None
        pop.add("a", fitness=3)
        pop.add("b", fitness=7)
        assert pop.best_fitness == 7

    def test_is_solved(self):
        pop = Population(task_id="t1")
        assert pop.is_solved is False
        pop.add("a", fitness=0)
        assert pop.is_solved is False
        pop.add("b", fitness=1)
        assert pop.is_solved is True

    def test_select_best(self):
        pop = Population(task_id="t1")
        assert pop.select_best() is None
        pop.add("a", fitness=3)
        pop.add("b", fitness=7)
        best = pop.select_best()
        assert best is not None
        assert best.code == "b"

    def test_indices_increment(self):
        pop = Population(task_id="t1")
        c1 = pop.add("a", fitness=1)
        c2 = pop.add("b", fitness=2)
        c3 = pop.add("c", fitness=3)
        assert c1.index == 0
        assert c2.index == 1
        assert c3.index == 2


# --- Population.select (tournament) ---


class TestPopulationSelect:
    def test_select_empty(self):
        pop = Population(task_id="t1")
        assert pop.select() is None

    def test_select_single(self):
        pop = Population(task_id="t1")
        c = pop.add("only", fitness=5)
        selected = pop.select(tournament_size=3)
        assert selected is not None
        assert selected.code == "only"

    def test_tournament_deterministic_with_seed(self):
        """Same seed produces same selection sequence."""
        rng1 = random.Random(99)
        rng2 = random.Random(99)

        pop1 = Population(task_id="t1", rng=rng1)
        pop2 = Population(task_id="t1", rng=rng2)

        for code, fit in [("a", 3), ("b", 7), ("c", 5), ("d", 1), ("e", 9)]:
            pop1.add(code, fitness=fit)
            pop2.add(code, fitness=fit)

        selections_1 = [pop1.select(tournament_size=3).code for _ in range(10)]
        selections_2 = [pop2.select(tournament_size=3).code for _ in range(10)]

        assert selections_1 == selections_2

    def test_tournament_returns_best_of_pool(self):
        pop = Population(task_id="t1", rng=random.Random(42))
        pop.add("a", fitness=1)
        pop.add("b", fitness=10)
        pop.add("c", fitness=5)

        # With tournament_size=3, all three are sampled, best is "b"
        selected = pop.select(tournament_size=3)
        assert selected.code == "b"

    def test_tournament_k_larger_than_pop(self):
        """When k > population size, samples entire population."""
        pop = Population(task_id="t1", rng=random.Random(42))
        pop.add("a", fitness=1)
        pop.add("b", fitness=10)

        selected = pop.select(tournament_size=5)
        assert selected.code == "b"


# --- Mutation hooks ---


class TestMutation:
    def _upper_hook(self, code: str) -> str:
        return code.upper()

    def test_mutate_empty_population(self):
        pop = Population(task_id="t1")
        result = pop.mutate(self._upper_hook, generation=1)
        assert result is None

    def test_mutate_adds_new_candidate(self):
        pop = Population(task_id="t1", rng=random.Random(42))
        pop.add("hello world", fitness=5)

        new_c = pop.mutate(self._upper_hook, generation=1)
        assert new_c is not None
        assert "HELLO" in new_c.code
        assert new_c.fitness == -1  # unevaluated sentinel
        assert new_c.generation == 1

    def test_mutate_best_adds_from_best(self):
        pop = Population(task_id="t1", rng=random.Random(42))
        pop.add("weak", fitness=1)
        pop.add("strong", fitness=9)

        new_c = pop.mutate_best(self._upper_hook, generation=2)
        assert new_c is not None
        assert "STRONG" in new_c.code

    def test_mutate_respects_max_size(self):
        pop = Population(task_id="t1", max_size=3, rng=random.Random(42))
        pop.add("a", fitness=5)
        pop.add("b", fitness=3)
        pop.add("c", fitness=7)

        new_c = pop.mutate(self._upper_hook, generation=1)
        assert new_c is not None
        assert len(pop.candidates) == 3  # evicted worst before adding


# --- SwarmPopulation ---


class TestSwarmPopulation:
    def test_get_or_create_new(self):
        swarm = SwarmPopulation(seed=42)
        pop = swarm.get_or_create("task_a")
        assert pop.task_id == "task_a"

    def test_get_or_create_existing(self):
        swarm = SwarmPopulation(seed=42)
        pop1 = swarm.get_or_create("task_a")
        pop2 = swarm.get_or_create("task_a")
        assert pop1 is pop2

    def test_population_for_missing(self):
        swarm = SwarmPopulation()
        assert swarm.population_for("nonexistent") is None

    def test_solved_and_unsolved_tasks(self):
        swarm = SwarmPopulation(seed=42)
        pa = swarm.get_or_create("a")
        pb = swarm.get_or_create("b")
        pc = swarm.get_or_create("c")

        pa.add("code", fitness=0)  # not solved (fitness == 0)
        pb.add("code", fitness=3)  # solved
        pc.add("code", fitness=1)  # solved

        assert "a" in swarm.unsolved_tasks
        assert "b" in swarm.solved_tasks
        assert "c" in swarm.solved_tasks

    def test_deterministic_across_swarm(self):
        """Two swarms with same seed produce identical selection results."""
        swarm1 = SwarmPopulation(seed=7)
        swarm2 = SwarmPopulation(seed=7)

        for tid in ["t1", "t2"]:
            pop1 = swarm1.get_or_create(tid)
            pop2 = swarm2.get_or_create(tid)
            for code, fit in [("a", 3), ("b", 7), ("c", 5)]:
                pop1.add(code, fitness=fit)
                pop2.add(code, fitness=fit)

        sel1 = [swarm1.get_or_create("t1").select().code for _ in range(5)]
        sel2 = [swarm2.get_or_create("t1").select().code for _ in range(5)]
        assert sel1 == sel2
