# Testing

Install dependencies with
```bash
pip install -r tests/requirements.txt
```

and run tests with
```
pytest tests/ -v
```

Sometimes tests with `ddp` fails because their are launched from the same process (pytest). Running them separately solves the issue, so the problem is not in this framework but in the conflicts between different `ddp` runs. `--forked` does not work because it is not supported by CUDA.