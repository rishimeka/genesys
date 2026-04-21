"""Allow running with: python -m genesys_memory"""
from genesys_memory.server import main
import asyncio

asyncio.run(main())
