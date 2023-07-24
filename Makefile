all: serve

pre-install:
	sudo apt update && sudo apt install -y python3-dev

serve:
	poetry run python api.py

requirements.txt: pyproject.toml
	poetry export --without-hashes -f requirements.txt --output requirements.txt