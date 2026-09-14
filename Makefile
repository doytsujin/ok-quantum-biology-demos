# Run and check both demos. Recipes need python3 with the pinned requirements.
PY ?= python3

.PHONY: help install quantum quantum-stage1 assembly classifier twin check clean

help:
	@echo "make install         pip install the pinned requirements of both folders"
	@echo "make check           run both demos and assert their verified outputs"
	@echo "make quantum         both quantum stages behind the gateway"
	@echo "make quantum-stage1  assembly and governance only"
	@echo "make assembly        QAOA assembly alone"
	@echo "make classifier      variational quantum classifier alone"
	@echo "make twin            bioprocess model, controller and decision artifact"

install:
	$(PY) -m pip install -r quantum-pipeline/requirements.txt -r bioprocess-twin/requirements.txt

quantum:
	cd quantum-pipeline && $(PY) governed_stage.py

quantum-stage1:
	cd quantum-pipeline && $(PY) governed_stage.py --stage1-only

assembly:
	cd quantum-pipeline && $(PY) quaser_assembly.py

classifier:
	cd quantum-pipeline && $(PY) qml_classifier.py

twin:
	cd bioprocess-twin && env -u ANTHROPIC_API_KEY $(PY) run_demo.py

check:
	cd quantum-pipeline && $(PY) governed_stage.py && $(PY) check.py
	cd bioprocess-twin && $(PY) check.py

clean:
	rm -rf quantum-pipeline/output */__pycache__
