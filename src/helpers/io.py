import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_folder_if_not_exists(folder_path):
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)


# def get_full_path_to_domino_dataset(dataset_name: str) -> Path:
#     """
#     Depending on if your project is Domino File System (DFS) based
#     or git-based, and if the dataset is associated with the given
#     project or shared from another project, the path to the dataset
#     is different.

#     Official documentation:
#     https://docs.dominodatalab.com/en/latest/user_guide/6942ab/use-datasets-and-snapshots/
#     """
#     paths = [
#         Path(f"/domino/datasets/local/{dataset_name}"),
#         Path(f"/domino/datasets/{dataset_name}"),
#         Path(f"/mnt/data/{dataset_name}"),
#         Path(f"/mnt/imported/data/{dataset_name}"),
#     ]

#     for path in paths:
#         if path.exists() and path.is_dir():
#             return path
#     raise IOError(
#         f"Domino datset '{dataset_name}' cannot be located. Please make sure it's mounted to your workspace"
#     )
