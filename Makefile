all: serve

pre-install:
	sudo apt update && sudo apt install -y python3-dev
	env -C python poetry install

serve:
	env -C python poetry run python api.py -m ../models/chatglm2-ggml.bin

python/requirements.txt: python/pyproject.toml
	env -C python poetry export --without-hashes -f requirements.txt --output requirements.txt