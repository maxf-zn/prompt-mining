"""
Enron mail corpus dataset from HuggingFace.

Benign business emails from Enron Corporation employees.
"""

from typing import Dict, Any, Optional
from datasets import load_dataset
from prompt_mining.datasets.hf_dataset import HFDataset
from prompt_mining.core.prompt_spec import PromptSpec


class EnronDataset(HFDataset):
    """
    Enron mail corpus mini dataset.

    Dataset: amanneo/enron-mail-corpus-mini

    Business emails from Enron Corporation employees. All examples are benign.

    Schema:
    - id: float (email ID)
    - email_type: str (email category)
    - text: str (email body content)
    - mail_length: int (message length)

    Splits:
    - train: 36k examples
    - test: 4k examples
    """

    def __init__(
        self,
        split: str = "train",
        name: Optional[str] = None,
        dataset_id: Optional[str] = None,
        include_email_format: bool = True,
        **kwargs
    ):
        """
        Initialize Enron dataset.

        Args:
            split: Dataset split to load ("train" or "test")
            name: Dataset name (default: "enron")
            dataset_id: Dataset ID (default: "enron")
            include_email_format: If True, format as "Email:\n{text}"
            **kwargs: Additional arguments passed to HFDataset (e.g., cache_dir)
        """
        self.include_email_format = include_email_format

        super().__init__(
            repo_id="amanneo/enron-mail-corpus-mini",
            split=split,
            name=name or "enron",
            dataset_id=dataset_id or "enron",
            **kwargs
        )

    def _load_hf_dataset(self):
        """
        Load dataset from HuggingFace by loading parquet directly.

        The enron-mail-corpus-mini dataset has a metadata/schema mismatch
        (metadata says 2 columns but parquet has 4). We bypass this by
        loading the parquet files directly without schema validation.
        """
        from huggingface_hub import HfApi, hf_hub_download
        from datasets import Dataset
        import pyarrow.parquet as pq

        # Find parquet file for requested split
        api = HfApi()
        files = api.list_repo_files(self.repo_id, repo_type="dataset", revision=self.revision)

        # Find parquet file matching split (e.g., "data/train-*.parquet")
        parquet_files = [f for f in files if f.startswith(f"data/{self.split}-") and f.endswith(".parquet")]
        if not parquet_files:
            raise ValueError(f"No parquet file found for split '{self.split}' in {self.repo_id}")

        # Download and concatenate all parquet files for the split
        # Only select columns we need (dataset has inconsistent schemas across files)
        columns = ["text", "mail_length"]
        tables = []
        for parquet_file in parquet_files:
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=parquet_file,
                repo_type="dataset",
                revision=self.revision,
                cache_dir=self.cache_dir,
            )
            table = pq.read_table(local_path)
            # Select only needed columns (handles schema inconsistency)
            available_cols = [c for c in columns if c in table.column_names]
            tables.append(table.select(available_cols))

        # Concatenate if multiple files
        import pyarrow as pa
        combined_table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        return Dataset(combined_table)

    def _convert_to_prompt_spec(
        self,
        example: Dict[str, Any],
        index: int
    ) -> PromptSpec:
        """
        Convert Enron example to PromptSpec.

        Args:
            example: Raw HF example with 'text' field
            index: Index in dataset

        Returns:
            PromptSpec with email content as user message
        """
        text = example.get("text", "")

        # Format prompt
        if self.include_email_format:
            content = f"Email:\n{text}"
        else:
            content = text

        messages = [
            {"role": "user", "content": content}
        ]

        # All Enron emails are benign business communications
        labels = {
            "malicious": False,
            "attack_type": None,
        }

        prompt_id = f"{self.dataset_id}:{self.split}:{index}"

        return PromptSpec(
            prompt_id=prompt_id,
            dataset_id=self.dataset_id,
            messages=messages,
            labels=labels,
            tools=None
        )
