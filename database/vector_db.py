import os
import pickle

class VectorDatabase:
    def __init__(self, db_path="database/user_data/embeddings.pkl"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.data = self._load_db()

    def _load_db(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                return pickle.load(f)
        return {}

    def save_user(self, user_id, vector):
        self.data[user_id] = vector
        with open(self.db_path, "wb") as f:
            pickle.dump(self.data, f)

    def get_user_vector(self, user_id):
        return self.data.get(user_id, None)

    def exists(self, user_id):
        return user_id in self.data