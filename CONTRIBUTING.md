# Contributing

Hey! Thanks for checking out this project. If you want to contribute, here's how:

## Getting Started

```bash
# Clone it
git clone https://github.com/YOUR_USERNAME/rag-evaluation-course.git
cd rag-evaluation-course

# Set up your environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install everything
pip install -e ".[dev]"

# Make sure tests work
pytest
```

## Making Changes

1. Fork the repo
2. Create a branch: `git checkout -b cool-new-feature`
3. Make your changes
4. Run tests: `pytest`
5. Commit: `git commit -m "Added cool feature"`
6. Push: `git push origin cool-new-feature`
7. Open a PR

## Code Style

- Keep it clean and readable
- Add type hints where it makes sense
- Write tests for new features
- Add docstrings to functions

## Need Help?

Just open an issue and ask! We're here to help.
