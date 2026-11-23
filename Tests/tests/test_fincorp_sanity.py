"""FINCORP dataset and notebook sanity checks (no model training required).

Run from repository root: python -m unittest discover -s Tests/tests -v
"""
import ast
import csv
import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / 'FINCORP_Reproduction'
IMPROVEMENT = ROOT / 'FINCORP_Improvment Experiments'
NOTEBOOK = ORIGINAL / 'MTL Model Experiments' / 'Quad_Task_(train_test_split).ipynb'
NEURAL_NOTEBOOK = IMPROVEMENT / 'code' / 'FINCORP_DistilRoBERTa_V1-V5.ipynb'
ORIGINAL_CSV = ORIGINAL / 'Database' / 'FINCORP.csv'
IMPROVEMENT_CSV = IMPROVEMENT / 'data' / 'FINCORP.csv'
COLUMNS = ['Domain', 'Complaint/ Opinion', 'Complaint Label', 'Severity level', 'Sentiment', 'Emotion']
SEVERITY = {'Non-Complaint', 'No explicit reproach', 'Disapproval', 'Accusation', 'Blame'}


def read_rows(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    return fields, rows


class TestFINCORPData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not ORIGINAL_CSV.is_file():
            raise AssertionError('Missing authors\' FINCORP.csv at: ' + str(ORIGINAL_CSV))
        cls.columns, cls.rows = read_rows(ORIGINAL_CSV)

    def test_data_dimensions_and_columns(self):
        self.assertEqual(self.columns, COLUMNS)
        self.assertEqual(len(self.rows), 6281, 'Released CSV has 6,281 data rows (paper reports 6,282)')
        self.assertTrue(all(None not in row for row in self.rows), 'Malformed CSV row')

    def test_no_empty_values(self):
        for idx, row in enumerate(self.rows, start=2):
            self.assertTrue(all(str(value).strip() for value in row.values()),
                            f'Empty entry in CSV line {idx}')

    def test_label_distributions(self):
        complaint = Counter(row['Complaint Label'].strip() for row in self.rows)
        severity = Counter(row['Severity level'].strip() for row in self.rows)
        self.assertEqual(complaint, {'1': 3149, '0': 3132})
        self.assertEqual(severity, {'Non-Complaint': 3132,
                                    'No explicit reproach': 1548,
                                    'Disapproval': 604,
                                    'Blame': 875,
                                    'Accusation': 122})
        self.assertEqual(set(severity), SEVERITY)
        self.assertEqual(len(set(row['Sentiment'].strip() for row in self.rows)), 3)
        self.assertEqual(len(set(row['Emotion'].strip() for row in self.rows)), 7)

    def test_complaint_severity_alignment(self):
        for idx, row in enumerate(self.rows, start=2):
            label = row['Complaint Label'].strip()
            severity = row['Severity level'].strip()
            self.assertEqual(label == '0', severity == 'Non-Complaint',
                             f'Inconsistent labels on CSV line {idx}')

    def test_improvement_data_matches_original(self):
        self.assertTrue(IMPROVEMENT_CSV.is_file(), f'Missing improvement CSV: {IMPROVEMENT_CSV}')
        self.assertEqual(hashlib.sha256(ORIGINAL_CSV.read_bytes()).digest(),
                         hashlib.sha256(IMPROVEMENT_CSV.read_bytes()).digest(),
                         'Authors\' and improvement CSVs have different bytes')


class TestProjectFiles(unittest.TestCase):
    def test_original_notebook_is_readable(self):
        self.assertTrue(NOTEBOOK.is_file(), f'Missing original notebook: {NOTEBOOK}')
        notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        self.assertEqual(notebook['nbformat'], 4)
        self.assertTrue(notebook.get('cells'))
        source = '\n'.join(''.join(cell.get('source', []))
                           for cell in notebook['cells'] if cell.get('cell_type') == 'code')
        self.assertEqual(source.count('model.fit('), 5,
                         'Expected five original model.fit blocks; notebook may have changed')

    def test_improvement_cpu_scripts_parse(self):
        code_dir = IMPROVEMENT / 'code'
        self.assertTrue(code_dir.is_dir(), f'Missing improvement code directory: {code_dir}')
        scripts = sorted(code_dir.glob('*.py'))
        self.assertGreaterEqual(len(scripts), 5, 'Expected five uploaded CPU experiment scripts')
        for script in scripts:
            with self.subTest(script=script.name):
                ast.parse(script.read_text(encoding='utf-8-sig'), filename=str(script))


    def test_neural_notebook_integrity_and_python_syntax(self):
        self.assertTrue(NEURAL_NOTEBOOK.is_file(),
                        f'Missing V1-V5 notebook: {NEURAL_NOTEBOOK}')
        notebook = json.loads(NEURAL_NOTEBOOK.read_text(encoding='utf-8'))
        self.assertEqual(notebook.get('nbformat'), 4)
        cells = notebook.get('cells', [])
        self.assertTrue(cells, 'V1-V5 notebook has no cells')
        code_cells = [cell for cell in cells if cell.get('cell_type') == 'code']
        self.assertGreaterEqual(len(code_cells), 7)
        for idx, cell in enumerate(code_cells, start=1):
            source = cell.get('source', [])
            self.assertIsInstance(source, (str, list))
            source = ''.join(source) if isinstance(source, list) else source
            # The installation command is a valid Colab/IPython magic, not Python AST.
            python_source = '\\n'.join(
                line for line in source.splitlines()
                if not line.lstrip().startswith(('%', '!'))
            )
            with self.subTest(cell=idx):
                ast.parse(python_source, filename=f'{NEURAL_NOTEBOOK.name}:cell-{idx}')

    def test_neural_notebook_covers_v1_to_v5(self):
        notebook = json.loads(NEURAL_NOTEBOOK.read_text(encoding='utf-8'))
        source = '\\n'.join(''.join(cell.get('source', []))
                           for cell in notebook['cells']
                           if cell.get('cell_type') == 'code')
        for experiment in ('V1', 'V2', 'V3', 'V4', 'V5'):
            with self.subTest(experiment=experiment):
                self.assertIn(f"run('{experiment}')", source,
                              f'Missing runnable {experiment} invocation')
        self.assertIn("def run(name):", source)
        self.assertIn("def train(", source)

    def test_neural_notebook_v3_and_v5_dependencies(self):
        notebook = json.loads(NEURAL_NOTEBOOK.read_text(encoding='utf-8'))
        source = '\\n'.join(''.join(cell.get('source', []))
                           for cell in notebook['cells']
                           if cell.get('cell_type') == 'code')
        self.assertIn("OUT/'V2'/'validation_predictions.csv'", source,
                      'V3 should load V2 validation predictions')
        self.assertIn('p[:,0]>=0.05', source,
                      'V3 should apply the recorded Accusation threshold')
        self.assertIn("checkpoint=train('V5_external_stage'", source,
                      'V5 should train the external severity model first')
        self.assertIn("return train('V5',tr,va,LABELS,starting=str(checkpoint))", source,
                      'V5 should transfer the external checkpoint')

if __name__ == '__main__':
    unittest.main()
