
.PHONY: all data teacher hidden logits gkd eval

export PYTHONPATH := $(CURDIR)

all: ## Run the full pipeline
	$(MAKE) data
	$(MAKE) teacher
	$(MAKE) hidden
	$(MAKE) logits
	$(MAKE) gkd
	$(MAKE) eval

data:
	python scripts/fetch_eastmoney.py --cfg configs/data.yaml
	python scripts/build_dataset.py --cfg configs/data.yaml

teacher:
	python scripts/run_teacher_infer.py --cfg configs/teacher_infer.yaml

hidden:
	bash scripts/train_hidden_kd.sh

logits:
	bash scripts/train_logits_kd.sh

gkd:
	bash scripts/train_gkd.sh

eval:
	python scripts/evaluate.py --cfg configs/eval.yaml
