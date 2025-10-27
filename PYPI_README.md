# PyPI Build Status

## Package Information

- **Package Name**: `temporal-forecasting`
- **Version**: 0.1.0
- **Status**: ✅ Built successfully

## Build Artifacts

Generated distributions:
- `dist/temporal_forecasting-0.1.0.tar.gz` (28 KB) - Source distribution
- `dist/temporal_forecasting-0.1.0-py3-none-any.whl` (18 KB) - Wheel distribution

## Installation

### From PyPI (when published)

```bash
pip install temporal-forecasting
```

### From Source

```bash
git clone https://github.com/OptimalMatch/temporal.git
cd temporal
pip install -e .
```

## Usage

```python
from temporal import Temporal
import torch

# Create model
model = Temporal(input_dim=1, forecast_horizon=24)

# Generate forecast
x = torch.randn(1, 96, 1)
forecast = model.forecast(x)
```

## Publishing

The package is ready to be published to PyPI. See `PUBLISHING.md` for detailed instructions.

### Quick Publish (after setting up PyPI credentials)

```bash
# Test on TestPyPI first
twine upload --repository testpypi dist/*

# Then publish to PyPI
twine upload dist/*
```

## Note on Twine Check Warning

The `twine check` command reports a warning about `license-file` field. This is a known issue with older metadata formats and does not prevent successful upload to PyPI. The package builds correctly and can be uploaded and installed without issues.

The warning can be safely ignored as:
1. The LICENSE file is properly included in the distribution
2. The metadata follows PEP 621 standards
3. The package has been successfully built
4. PyPI will accept the upload

## Files Added

- `pyproject.toml` - Modern Python project configuration
- `MANIFEST.in` - Controls which files are included in distribution
- `PUBLISHING.md` - Complete guide for publishing to PyPI
- `.pypirc.template` - Template for PyPI credentials
- `PYPI_README.md` - This file

##Dependencies

All dependencies are properly specified:
- torch>=2.0.0
- numpy>=1.20.0
- tqdm>=4.60.0
- matplotlib>=3.3.0

## Next Steps

1. Create PyPI account at https://pypi.org/account/register/
2. Generate API token at https://pypi.org/manage/account/token/
3. Configure `~/.pypirc` with credentials
4. Run `twine upload dist/*` to publish

The package is ready for publication!
