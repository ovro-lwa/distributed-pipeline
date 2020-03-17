set -e

echo 'Running pytest...'
pytest --cov=orca
echo 'Running mypy for type check...'
mypy --ignore-missing-imports orca
