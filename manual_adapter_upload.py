from huggingface_hub import HfApi, create_repo, upload_file

# Define the path to your files
adapter_config_path = "aicvd/adapter_config.json"
adapter_model_path = "aicvd/adapter_model.safetensors"
repo_name = "automateyournetwork/aicvd"

# Initialize the HfApi client
api = HfApi()

# Ensure the repository exists
create_repo(repo_name, exist_ok=True)

# Upload files individually
api.upload_file(
    path_or_fileobj=adapter_config_path,
    path_in_repo="adapter_config.json",
    repo_id=repo_name,
    repo_type="model"
)

api.upload_file(
    path_or_fileobj=adapter_model_path,
    path_in_repo="adapter_model.safetensors",
    repo_id=repo_name,
    repo_type="model"
)
