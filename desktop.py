
import threading
import time
import webview
from main import app


def run_flask():
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False
    )


if __name__ == "__main__":
    server = threading.Thread(target=run_flask, daemon=True)
    server.start()

    time.sleep(1)

    webview.create_window(
        "Signature Identification System",
        "http://127.0.0.1:5000",
        width=1200,
        height=800,
        min_size=(900, 600)
    )

    webview.start()
