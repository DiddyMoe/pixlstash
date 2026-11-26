"""Work outside a request that holds a library read lease, as a request does.

The smart-set refresher, an Open-with edit-copy pass and an export run on
threads of their own, so no admission middleware hands them a lease. Each
takes one from ``LibraryGenerationCoordinator`` and runs against the vault
that lease names, through :class:`PinnedVaultServer`; a library switch or a
full restore waits for the lease, so the work never writes into a library
file that replaced the one it read.
"""

#: How many times such work tries again, one wait apart, while admission is
#: READY but no coherent library is open (``LibraryNotReadyError`` with state
#: UNAVAILABLE), before it gives up until its next trigger. That state can
#: last until a restart, so the tries are bounded; a switch or a restore
#: (SWITCHING) always ends, so it is waited out without a limit.
UNAVAILABLE_RETRIES = 5


class PinnedVaultServer:
    """The server, with ``vault`` fixed to one leased vault.

    Everything else (settings, the hub, the coordinator) is the server's own,
    so code written against the server reads the leased vault through
    ``server.vault`` for the whole pass.
    """

    def __init__(self, server, vault):
        self._server = server
        self.vault = vault

    def __getattr__(self, name):
        return getattr(self._server, name)
