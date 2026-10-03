import glob
import json
from pathlib import Path

rows = []
for log_path in glob.glob("data/raw/SBOX/*/log.jsonl"):
    run_id = Path(log_path).parent.name  
    with open(log_path) as f:
        for line in f:
            record = json.loads(line)
            record["run"] = run_id
            rows.append(record)

print(len(rows))
