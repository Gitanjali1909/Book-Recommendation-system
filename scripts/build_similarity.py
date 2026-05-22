import os
import pickle

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


with open("outputs/features.pkl", "rb") as f:
    features = pickle.load(f)

similarity_matrix = cosine_similarity(features).astype(np.float32)

os.makedirs("outputs", exist_ok=True)

with open("outputs/similarity_matrix.pkl", "wb") as f:
    pickle.dump(similarity_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)

print("Similarity matrix saved:", similarity_matrix.shape)
