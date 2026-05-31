"""Allows  python -m outcast_virus  from repo root."""
import asyncio
from .orchestrator import main

asyncio.run(main())
