class ModClientNotConnected(Exception):
    def __init__(self, message="Mod client disconnected."):
        super().__init__(message)

class ModClientAlreadyConnected(Exception):
    def __init__(self, message="Only one mod client can be connected at a time."):
        super().__init__(message)