"""Allow running with: python -m genesys.server"""
from genesys.server import main
import asyncio

asyncio.run(main())
