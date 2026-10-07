"""Check the published evaluation accounting without downloading model weights."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'eval/val_large.json').read_text(encoding='utf-8'))
matrix=json.loads((ROOT/'evaluation/router-confusion.json').read_text())
summary=json.loads((ROOT/'evaluation/router_run.json').read_text())
assert len({x['id'] for x in data})==len(data)==summary['samples']==340
assert sum(sum(row) for row in matrix['counts'])==len(data)
assert sum(matrix['counts'][i][i] for i in range(4))==summary['correct']==296
assert abs(summary['correct']/summary['samples']-summary['accuracy'])<1e-12
assert {x['category'] for x in data}==set(matrix['labels'])
print('340 unique questions; confusion matrix and accuracy are internally consistent.')
