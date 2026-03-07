PYTHON = uv run python3
UV = uv
MAIN_FILES = main.py

all: install run

install:
	@echo "executing (LLM) Call_Me_Maybe"
	$(UV) sync

run:
	@echo "executing (LLM) Call_Me_Maybe"
	$(PYTHON) $(MAIN_FILES) $(CONFIG_FILE)

clean:
	@echo "remove files"
	rm -rf __pycache__ .venv .uv

lint:
	@echo "check code quality (--strict - mode)"
	flake8 . --exclude .venv
	mypy . --strict --warn-return-any \
	--warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs \
	--check-untyped-defs
