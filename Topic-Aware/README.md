# The Topic-Aware Memory Bank in PALACE

This module will handle the `"context"` field in the dataset, retrieving and sorting the dialogue history from previous sessions, and training the `topic detector` to process the dialogue history of the current session.

## 1. Train topic detector

You can run the script to train the topic detector and save its checkpoint:

`bash topic_detector/topic_detector.sh`

## 2. Retrieval long-term history and re-rank

You can run the script below to obtain topic-related dialogue history and save the processed dataset with the dialogue history for subsequent training.

`bash Memroy_bank/memory_bank.bash`
