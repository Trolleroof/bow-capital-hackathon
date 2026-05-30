"""Allows  python -m combatos  from repo root."""
import asyncio
from .orchestrator import main

asyncio.run(main())
