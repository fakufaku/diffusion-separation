import os
from pathlib import Path


class SplitDirectory:
    """
    Split content of directory into smaller directories
    with a maximum number of files
    """

    def __init__(self, root, n_files_per_dir=1000, n_digits=4):
        self.root = Path(root)
        self.n_files_per_dir = n_files_per_dir
        self.n_digits = n_digits
        self.dir_idx = 0
        self.file_idx = 0
        self._make_subdir()

    def _make_subdir(self):
        while True:
            self.subdir = self.root / f"{self.dir_idx:0{self.n_digits}d}"
            if self.subdir.exists():
                # check the current number of files
                n_files = len(os.listdir(self.subdir))
                if n_files < self.n_files_per_dir:
                    self.file_idx = n_files
                    break
                else:
                    # directory is full, increment and loop
                    self.dir_idx += 1
            else:
                self.subdir.mkdir(exist_ok=True, parents=True)
                self.file_idx = 0
                break

    def get_path(self, basename, suffix):

        # if we reach max number of files, make new dir
        if self.file_idx == self.n_files_per_dir:
            self.dir_idx += 1
            self.file_idx = 0
            self._make_subdir()

        path = (self.subdir / basename).with_suffix(suffix)
        self.file_idx += 1

        return str(path)
