fmt:
	poetry run black . --preview

lint:
	poetry run ruff check .

lint-fix:
	poetry run ruff check . --fix
