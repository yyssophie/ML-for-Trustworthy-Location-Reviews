import kagglehub

# Download latest version
path = kagglehub.dataset_download("denizbilginn/google-maps-restaurant-reviews")

print("Path to dataset files:", path)