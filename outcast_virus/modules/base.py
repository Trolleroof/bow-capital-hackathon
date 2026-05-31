"""AbstractModule — interface every Outcast Virus module wrapper must implement."""
from __future__ import annotations
import abc


class AbstractModule(abc.ABC):
    name: str  # must be one of: nav | perception | recon | swarm

    @abc.abstractmethod
    async def run(self) -> None:
        """Main coroutine.  Should loop forever; the orchestrator restarts it on crash."""
        ...
