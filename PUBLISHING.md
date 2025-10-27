# Publishing to PyPI

This guide explains how to publish the Temporal package to PyPI.

## Prerequisites

1. **Install build tools**:
   ```bash
   pip install --upgrade pip build twine
   ```

2. **Create PyPI account**:
   - Main PyPI: https://pypi.org/account/register/
   - Test PyPI: https://test.pypi.org/account/register/

3. **Generate API tokens**:
   - PyPI: https://pypi.org/manage/account/token/
   - Test PyPI: https://test.pypi.org/manage/account/token/

4. **Configure credentials**:
   ```bash
   # Copy template
   cp .pypirc.template ~/.pypirc

   # Edit ~/.pypirc and add your API tokens
   nano ~/.pypirc

   # Secure the file
   chmod 600 ~/.pypirc
   ```

## Build the Package

1. **Clean previous builds**:
   ```bash
   rm -rf build/ dist/ *.egg-info
   ```

2. **Build distribution packages**:
   ```bash
   python -m build
   ```

   This creates:
   - `dist/temporal-forecasting-0.1.0.tar.gz` (source distribution)
   - `dist/temporal_forecasting-0.1.0-py3-none-any.whl` (wheel distribution)

3. **Verify the build**:
   ```bash
   twine check dist/*
   ```

## Test on TestPyPI (Recommended)

1. **Upload to TestPyPI**:
   ```bash
   twine upload --repository testpypi dist/*
   ```

2. **Install from TestPyPI**:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ \
       temporal
   ```

3. **Test the installation**:
   ```bash
   python -c "from temporal import Temporal; print('Success!')"
   ```

## Publish to PyPI

1. **Upload to PyPI**:
   ```bash
   twine upload dist/*
   ```

2. **Verify on PyPI**:
   - Visit: https://pypi.org/project/temporal/

3. **Install from PyPI**:
   ```bash
   pip install temporal
   ```

## Version Management

### Update Version

Edit `pyproject.toml`:
```toml
[project]
version = "0.1.1"  # Increment version
```

### Versioning Guidelines

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR** version (1.0.0): Incompatible API changes
- **MINOR** version (0.1.0): Add functionality (backwards-compatible)
- **PATCH** version (0.0.1): Bug fixes (backwards-compatible)

### Release Process

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG** (if you have one)
3. **Commit changes**:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 0.1.1"
   git tag v0.1.1
   git push && git push --tags
   ```
4. **Build and publish** (steps above)

## Quick Commands Reference

```bash
# Full release workflow
rm -rf build/ dist/ *.egg-info
python -m build
twine check dist/*
twine upload --repository testpypi dist/*  # Test first
twine upload dist/*                        # Then production

# Install locally for testing
pip install -e .

# Install from PyPI
pip install temporal

# Install specific version
pip install temporal==0.1.0

# Upgrade to latest
pip install --upgrade temporal
```

## Troubleshooting

### Issue: "File already exists"
If you try to upload the same version twice:
```bash
# Increment version in pyproject.toml
# Rebuild and upload
```

### Issue: "Invalid credentials"
Check `~/.pypirc` has correct API tokens:
```bash
cat ~/.pypirc
chmod 600 ~/.pypirc
```

### Issue: "Package dependencies not found"
Ensure dependencies are available on PyPI (not local packages).

### Issue: "README not rendering"
- Ensure README.md uses standard Markdown
- Check for syntax errors
- Test locally with: `python -m readme_renderer README.md`

## Package Information

- **Package name**: `temporal`
- **Import name**: `temporal`
- **PyPI URL**: https://pypi.org/project/temporal/
- **Repository**: https://github.com/OptimalMatch/temporal

## Security Notes

1. **Never commit** `.pypirc` or API tokens to git
2. **Use API tokens** instead of passwords
3. **Restrict token scope** to specific projects
4. **Rotate tokens** regularly
5. **Enable 2FA** on PyPI account

## Support

For issues:
- GitHub: https://github.com/OptimalMatch/temporal/issues
- PyPI Help: https://pypi.org/help/
