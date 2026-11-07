import os
import tempfile

import pytest

# Set BEFORE visionguard is imported: config.py reads the environment at import time.
# Tests use a throw-away SQLite file, never a real database or a real .env.
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(tempfile.mkdtemp(), "test.db")


@pytest.fixture(scope="session")
def client():
    from visionguard import create_app  # imported here, after DATABASE_URL is set

    # Loads the real champion.keras once (run pytest from the project root).
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()
