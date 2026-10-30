"""Opt-in Uvicorn launcher with cancellation before network handler draining.

Use ``python -m minillm.continuous_server``. Generic Uvicorn's lifespan shutdown
happens after waiting for open SSE handlers and cannot provide this ordering.
"""

import uvicorn


class ContinuousServer(uvicorn.Server):
    async def shutdown(self, sockets=None):
        app = self.config.app
        hook = getattr(getattr(app, "state", None), "begin_shutdown", None)
        if hook is not None:
            await hook()
        await super().shutdown(sockets=sockets)


def main():
    from .server import app, settings
    # Pass the actual app so the public shutdown hook is available. One worker
    # only: multiple workers would each own their own model and admission queue.
    ContinuousServer(uvicorn.Config(app, host=settings.host, port=settings.port,
        workers=1)).run()


if __name__ == "__main__":
    main()
