# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-02

### Added

#### Project Structure
- Reorganized codebase with standard `src/` layout
- Created `docs/` directory for documentation
- Created `examples/` directory for usage examples
- Created `deployment/` directory for deployment configurations
- Created `scripts/` directory for build and deployment scripts

#### Documentation
- Professional README.md with badges, features, and comprehensive usage instructions
- CONTRIBUTING.md with detailed contribution guidelines
- LICENSE file (MIT License)
- API documentation in `docs/API.md`
- Deployment guide in `docs/DEPLOYMENT.md`
- Examples README with usage instructions
- Comprehensive docstrings for all modules and functions

#### Code Quality
- Added type hints to utility functions
- Improved function naming (renamed `parse_arge` to `parse_args`)
- Added comprehensive docstrings following Google style
- Created `.pre-commit-config.yaml` for code quality checks
- Added `pyproject.toml` for tool configuration

#### Development Tools
- GitHub Actions CI/CD workflow for testing and linting
- GitHub Actions workflow for PyPI publishing
- Pre-commit hooks configuration
- Black, isort, flake8, and mypy configurations

#### Requirements
- Split requirements into three files:
  - `requirements.txt` - Core dependencies
  - `requirements-dev.txt` - Development dependencies
  - `requirements-deploy.txt` - Deployment dependencies

#### Examples
- `simple_inference.py` - Basic single-text classification
- `batch_inference.py` - Efficient batch processing
- `evaluation.py` - Complete evaluation workflow

#### Package Configuration
- Updated `setup.py` with proper metadata and entry points
- Added `pyproject.toml` for modern Python packaging
- Configured package for PyPI distribution

### Changed
- Moved all source code to `src/gpt_detector/` directory
- Moved training scripts to `src/gpt_detector/training/`
- Moved configuration to `src/gpt_detector/config/`
- Moved Docker files to `deployment/docker/`
- Moved Terraform files to `deployment/terraform/`
- Moved shell scripts to `scripts/`
- Updated all import statements to use new package structure
- Improved `.gitignore` with comprehensive exclusions

### Improved
- Code documentation with detailed docstrings
- Error handling in Flask application
- Configuration management
- Project organization and structure
- Developer experience with better tooling

### Fixed
- Typo in function name (`parse_arge` → `parse_args`)
- Import paths throughout the codebase
- Configuration file paths

## [Unreleased]

### Planned Features
- Multi-language support
- Model explainability features
- Browser extension
- Specific AI model detection (GPT-3, GPT-4, Claude, etc.)
- Batch processing API endpoint
- Model performance metrics dashboard
- Public demo website

---

## Version History

- **0.1.0** (2025-11-02) - Initial organized release
  - Complete project restructuring
  - Comprehensive documentation
  - Production-ready deployment options
  - CI/CD pipeline

---

For more details on each release, see the [GitHub Releases](https://github.com/yourusername/gpt-detector/releases) page.

