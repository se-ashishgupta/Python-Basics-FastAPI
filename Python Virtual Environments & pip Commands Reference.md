# Python Virtual Environments & pip Commands Reference

## 📁 Creating Virtual Environments

### Basic Creation

```bash
# Create virtual environment with default Python
python -m venv venv

# Create with custom name
python -m venv myproject_env

# Create with specific Python version
python3.9 -m venv venv
python3.10 -m venv venv

# Create without pip (rare use case)
python -m venv venv --without-pip

# Create with system site packages access
python -m venv venv --system-site-packages

# Clear existing environment and recreate
python -m venv venv --clear
```

## 🔌 Activating Virtual Environments

### Windows

```cmd
# Command Prompt (CMD)
venv\Scripts\activate

# PowerShell
venv\Scripts\Activate.ps1

# Git Bash
source venv/Scripts/activate
```

### Linux / macOS

```bash
# Bash/Zsh
source venv/bin/activate

# Fish shell
source venv/bin/activate.fish

# Csh/Tcsh
source venv/bin/activate.csh
```

### Verification

```bash
# Check if virtual environment is active
which python
# or
where python

# Check Python version
python --version

# Check pip location
which pip
```

## 📦 pip Package Management

### Installing Packages

```bash
# Install single package
pip install package_name

# Install specific version
pip install package_name==1.2.3

# Install minimum version
pip install package_name>=1.2.0

# Install from requirements file
pip install -r requirements.txt

# Install in development mode (for local packages)
pip install -e .

# Install from GitHub
pip install git+https://github.com/user/repo.git

# Install from specific branch
pip install git+https://github.com/user/repo.git@branch_name

# Install with extras
pip install package_name[extra1,extra2]

# Install without dependencies
pip install package_name --no-deps

# Upgrade package
pip install --upgrade package_name

# Upgrade pip itself
python -m pip install --upgrade pip
```

### Uninstalling Packages

```bash
# Uninstall single package
pip uninstall package_name

# Uninstall multiple packages
pip uninstall package1 package2 package3

# Uninstall from requirements file
pip uninstall -r requirements.txt

# Uninstall with confirmation skip
pip uninstall package_name -y
```

### Package Information

```bash
# List installed packages
pip list

# List outdated packages
pip list --outdated

# Show package details
pip show package_name

# Check for security vulnerabilities
pip audit

# List packages in requirements format
pip freeze

# List only user-installed packages
pip list --user

# List packages not required by others
pip list --not-required
```

## 📋 Requirements Management

### Creating Requirements Files

```bash
# Standard requirements
pip freeze > requirements.txt

# Development requirements
pip freeze > requirements-dev.txt

# Production requirements (manual creation recommended)
# Create requirements-prod.txt with only necessary packages

# Generate requirements from imports (requires pipreqs)
pip install pipreqs
pipreqs . --force
```

### Installing from Requirements

```bash
# Install from requirements
pip install -r requirements.txt

# Install with index URL
pip install -r requirements.txt -i https://pypi.org/simple/

# Install ignoring installed packages
pip install -r requirements.txt --force-reinstall

# Install only if requirements are newer
pip install -r requirements.txt --upgrade
```

### Requirements File Format Examples

```txt
# requirements.txt examples
Django==4.2.0
requests>=2.28.0
numpy==1.24.3
pandas>=1.5.0,<2.0.0

# With comments
Flask==2.3.0  # Web framework
SQLAlchemy>=1.4.0  # Database ORM

# From GitHub
git+https://github.com/user/repo.git@v1.0.0

# With extras
requests[security,socks]
```

## 🔍 Advanced pip Commands

### Cache Management

```bash
# Show cache info
pip cache info

# Clear all cache
pip cache purge

# Remove specific package from cache
pip cache remove package_name

# Show cache directory
pip cache dir
```

### Configuration

```bash
# Show configuration
pip config list

# Set configuration
pip config set global.index-url https://pypi.org/simple/

# Show configuration files location
pip config debug
```

### Searching and Dependencies

```bash
# Search packages (deprecated, use web)
pip search package_name

# Check dependencies
pip check

# Show dependency tree (requires pipdeptree)
pip install pipdeptree
pipdeptree

# Show reverse dependencies
pipdeptree -r
```

## 🛠️ Environment Management

### Environment Information

```bash
# Show environment info
python -m site

# List site packages
python -c "import site; print(site.getsitepackages())"

# Show Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### Deactivating Environment

```bash
# Deactivate current environment
deactivate

# Force deactivate (if normal doesn't work)
source deactivate  # Some systems
```

### Removing Virtual Environments

```bash
# Simply delete the folder
rm -rf venv/           # Linux/Mac
rmdir /s venv\         # Windows CMD
Remove-Item -Recurse -Force venv\  # PowerShell
```

## 🔧 Troubleshooting Commands

### Common Issues

```bash
# Fix broken pip
python -m ensurepip --upgrade

# Reinstall pip
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py

# Check Python executable
python -c "import sys; print(sys.executable)"

# Verify virtual environment
python -c "import sys; print(sys.prefix != sys.base_prefix)"

# Fix permission issues (use with caution)
pip install --user package_name
```

### Environment Variables

```bash
# Windows - Set Python path
set PYTHONPATH=%PYTHONPATH%;C:\path\to\your\project

# Linux/Mac - Set Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/your/project"

# Disable pip version check
export PIP_DISABLE_PIP_VERSION_CHECK=1
```

## 📚 Best Practices Checklist

### Project Setup

- [ ] Create virtual environment: `python -m venv venv`
- [ ] Activate environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
- [ ] Upgrade pip: `python -m pip install --upgrade pip`
- [ ] Install packages: `pip install package_name`
- [ ] Create requirements: `pip freeze > requirements.txt`
- [ ] Add `venv/` to `.gitignore`

### Daily Workflow

- [ ] Activate environment before working
- [ ] Install new packages as needed
- [ ] Update requirements file regularly
- [ ] Deactivate when switching projects

### Sharing Projects

- [ ] Include `requirements.txt` in repository
- [ ] Document Python version requirement
- [ ] Provide setup instructions in README
- [ ] Never commit virtual environment folder

## 🎯 Quick Reference Commands

| Action               | Command                              |
| -------------------- | ------------------------------------ |
| Create venv          | `python -m venv venv`                |
| Activate (Win)       | `venv\Scripts\activate`              |
| Activate (Unix)      | `source venv/bin/activate`           |
| Install package      | `pip install package_name`           |
| List packages        | `pip list`                           |
| Save requirements    | `pip freeze > requirements.txt`      |
| Install requirements | `pip install -r requirements.txt`    |
| Upgrade package      | `pip install --upgrade package_name` |
| Uninstall package    | `pip uninstall package_name`         |
| Deactivate           | `deactivate`                         |

## 🌐 Alternative Virtual Environment Tools

### conda

```bash
# Create environment
conda create -n myenv python=3.9

# Activate
conda activate myenv

# Install packages
conda install package_name

# Export environment
conda env export > environment.yml

# Create from file
conda env create -f environment.yml
```

### poetry

```bash
# Initialize project
poetry init

# Install dependencies
poetry install

# Add package
poetry add package_name

# Activate shell
poetry shell
```

### pipenv

```bash
# Create Pipfile and virtual environment
pipenv install

# Install package
pipenv install package_name

# Activate shell
pipenv shell

# Install from Pipfile
pipenv install
```

---

**💡 Pro Tip**: Always activate your virtual environment before installing packages, and remember to deactivate when switching between projects!
