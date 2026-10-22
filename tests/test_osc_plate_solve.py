"""Sequence-solve regression tests without importing Qt or connecting to Siril.

Run with: python -m unittest discover -s tests -v
Requires numpy, already a preprocessor dependency.
"""
import ast
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np


class SirilError(Exception):
    pass


SCRIPT = Path(__file__).resolve().parents[1] / 'Naztronomy-OSC_PP.py'
namespace = {
    'Path': Path, 'np': np,
    's': types.SimpleNamespace(SirilError=SirilError, DataError=SirilError,
                               CommandError=SirilError),
    'LogColor': types.SimpleNamespace(SALMON=0, GREEN=0),
}
methods = [node for cls in ast.parse(SCRIPT.read_text()).body
           if isinstance(cls, ast.ClassDef) for node in cls.body
           if isinstance(node, ast.FunctionDef) and node.name in
           ('seq_plate_solve', 'restore_plate_solve_selection')]
for method in methods:
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 str(SCRIPT), 'exec'), namespace)
Worker = type('Worker', (), {method.name: namespace[method.name] for method in methods})


class FakeSiril:
    def __init__(self, directory, initial, outcomes):
        self.directory = directory
        self.path = directory / 'lights_.seq'
        self.included = set(initial)
        self.outcomes = iter(outcomes)
        self.commands = []
        self.logs = []
        self.matrices = {}
        self.write()

    def write(self):
        lines = [f"S 'lights_' 10 5 {len(self.included)} 5 0 6 0 0 0", 'L 1']
        # FITS suffixes deliberately differ from one-based sequence positions.
        lines += [f'I {i + 9} {int(i in self.included)}' for i in range(1, 6)]
        lines += ['R0 1 1 1 0 0 10 H ' + self.matrices.get(i, '0 0 0 0 0 0 0 0 0')
                  for i in range(1, 6)]
        self.path.write_text('\n'.join(lines) + '\n')

    def get_siril_wd(self):
        return str(self.directory)

    def log(self, message, *_):
        self.logs.append(message)

    def cmd(self, *args):
        self.commands.append(args)
        if args[0] == 'seqplatesolve':
            included, matrices, error = next(self.outcomes)
            self.included = set(included)
            self.matrices = matrices
            self.write()
            if error:
                raise SirilError('solver failure')
        elif args[0] in ('select', 'unselect'):
            positions = set(range(args[2], args[3] + 1))
            assert positions <= set(range(1, 6))
            if args[0] == 'select':
                self.included |= positions
            else:
                self.included -= positions
            self.write()
        else:
            raise AssertionError(args)


def solution(indices, error=False):
    return set(indices), {i: '1 0 0 0 1 0 0 0 1' for i in indices}, error


class SequenceSolveTests(unittest.TestCase):
    def run_case(self, initial, outcomes, fallback=False):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        fake = FakeSiril(Path(temp.name), initial, outcomes)
        worker = Worker()
        worker.siril = fake
        worker.astrometry_net_available = True
        result = worker.seq_plate_solve('lights_', fallback)
        return result, fake

    def test_partial_success_keeps_registration_without_retry(self):
        result, fake = self.run_case(range(1, 6), [solution({1, 3, 4})], True)
        self.assertTrue(result)
        self.assertEqual(fake.included, {1, 3, 4})
        self.assertEqual(sum(c[0] == 'seqplatesolve' for c in fake.commands), 1)
        self.assertTrue(any('3/5' in line for line in fake.logs))

    def test_full_success(self):
        result, fake = self.run_case(range(1, 6), [solution(range(1, 6))])
        self.assertTrue(result)
        self.assertEqual(fake.included, set(range(1, 6)))

    def test_two_frames_are_sufficient(self):
        result, _ = self.run_case(range(1, 6), [solution({2, 5})])
        self.assertTrue(result)

    def test_one_frame_is_not_a_stack(self):
        result, fake = self.run_case(range(1, 6), [solution({1})])
        self.assertFalse(result)
        self.assertEqual(fake.included, set(range(1, 6)))

    def test_null_matrices_are_not_success(self):
        result, _ = self.run_case(range(1, 6), [(set(range(1, 6)), {}, False)])
        self.assertFalse(result)

    def test_nonfinite_singular_and_incomplete_matrices_are_excluded(self):
        matrices = {1: '1 0 0 0 1 0 0 0 1', 2: '1 0 0 0 1 0 0 0 1',
                    3: 'nan 0 0 0 1 0 0 0 1', 4: '1 0 0 1 0 0 0 0 1', 5: '1 0 0'}
        result, fake = self.run_case(range(1, 6), [(set(range(1, 6)), matrices, False)])
        self.assertTrue(result)
        self.assertEqual(fake.included, {1, 2})

    def test_prior_exclusions_are_preserved(self):
        result, fake = self.run_case({1, 3, 4}, [solution(range(1, 6))])
        self.assertTrue(result)
        self.assertEqual(fake.included, {1, 3, 4})

    def test_command_failure_cannot_claim_success_from_selection(self):
        result, _ = self.run_case(range(1, 6), [solution(range(1, 6), True)])
        self.assertFalse(result)

    def test_retry_restores_original_selection_and_omits_force(self):
        result, fake = self.run_case({1, 3, 4}, [solution({1}, True), solution({1, 3, 4})], True)
        self.assertTrue(result)
        solves = [c for c in fake.commands if c[0] == 'seqplatesolve']
        self.assertIn('-force', solves[0])
        self.assertNotIn('-force', solves[1])
        self.assertIn('-localasnet', solves[1])
        self.assertIn(('select', 'lights_', 3, 4), fake.commands)
        self.assertEqual(fake.included, {1, 3, 4})

    def test_failed_retries_restore_selection_for_global_registration(self):
        result, fake = self.run_case({1, 3, 4}, [solution({1}, True)] * 3, True)
        self.assertFalse(result)
        self.assertEqual(fake.included, {1, 3, 4})
        solves = [c for c in fake.commands if c[0] == 'seqplatesolve']
        self.assertIn('-blindpos', solves[-1])
        self.assertIn('-blindres', solves[-1])
        self.assertNotIn('-force', solves[-1])

    def test_empty_selection_does_not_invoke_solver(self):
        result, fake = self.run_case(set(), [], True)
        self.assertFalse(result)
        self.assertFalse(fake.commands)


if __name__ == '__main__':
    unittest.main()
