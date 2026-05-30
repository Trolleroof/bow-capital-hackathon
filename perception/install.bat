@echo off
:: Install norfair's optional dep (pure Python, no wheel needed) first
:: so the main install can proceed without the filterpy conflict.
python -m pip install "filterpy>=1.4.5,<2.0.0" --only-binary :all: 2>nul || python -m pip install "filterpy>=1.4.5,<2.0.0"

:: Install norfair without letting pip get blocked by its numpy<2.0 constraint
python -m pip install norfair --no-deps --only-binary :all:

:: Install everything else with binary wheels (norfair excluded from requirements.txt
:: to avoid its numpy<2.0 constraint poisoning the resolver)
python -m pip install -r requirements.txt --only-binary :all:

echo.
echo All dependencies installed.
