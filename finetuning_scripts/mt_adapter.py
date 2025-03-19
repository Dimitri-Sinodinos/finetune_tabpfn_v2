import torch
import torch.nn as nn

class MultiTaskAdapter(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        self.num_tasks = num_tasks

        # Adapter to map single output back to multiple tasks
        self.adapter = nn.Sequential(
            nn.Linear(5000, num_tasks),
            nn.GELU(),
            nn.Linear(num_tasks, num_tasks)
        )

    def forward(self, x):
        """
        Use an adapter to map TabPFN's output back to `num_tasks`.
        """

        # Map TabPFN single output to multi-task output
        multitask_output = self.adapter(x)
        return multitask_output
