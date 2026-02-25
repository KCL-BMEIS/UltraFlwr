"""
Dynamic Client that switches between SSL and regular training based on environment variable.
"""

import os
from flwr.client import ClientApp

# Check if we're in SSL mode
SSL_MODE = os.getenv("FEDYOLO_SSL_MODE", "false").lower() == "true"

if SSL_MODE:
    # Use SSL client
    from fedyolo.train.ssl_client import client_fn as ssl_client_fn

    client_fn = ssl_client_fn
else:
    # Use regular client
    from fedyolo.train.client import client_fn as regular_client_fn

    client_fn = regular_client_fn

# Create the app
app = ClientApp(client_fn)
