PYTHON ?= python3
PROJECT ?= MirrorOPD-verl
RUN ?=
READER = .venv/bin/python tools/wandb_local.py

.PHONY: setup list decode refresh csv test

setup:
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install -r requirements.txt

list:
	$(READER) list --project "$(PROJECT)"

decode:
	$(READER) decode --project "$(PROJECT)" $(if $(RUN),--run "$(RUN)",)

refresh:
	git pull --ff-only
	$(MAKE) decode

csv:
	@test -n "$(RUN)" || { echo 'Specify RUN=<run-id>'; exit 1; }
	$(READER) csv --project "$(PROJECT)" --run "$(RUN)"

test:
	.venv/bin/python -m unittest discover -s tests -v
