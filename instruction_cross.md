# Assignment: Implement a Cross-Encoder for Multi-label Classification

**Objective**  
You will implement a Cross-Encoder that takes multiple labels and a text, concatenates them in a single input in the format:  
```
[CLS] label_1 [CLS] label_2 ... [CLS] label_k [SEP] text [SEP]
```
and produces a separate classification score for each label.

The Cross-Encoder approach allows the transformer to attend across both the text and all labels in one forward pass. This typically gives a stronger modeling capacity compared to encoding the text and labels separately (as in a Bi-Encoder).  

**What You Will Do**  
1. **Build custom inputs** where each label is preceded by a `[CLS]` token.  
2. **Identify** the positions in the sequence that correspond to each label’s `[CLS]`.  
3. **Extract embeddings** for each label.  
4. **Pass** those embeddings through a classifier layer to generate label-specific scores.  
5. **Implement** a prediction function that returns readable outputs (label → score).

---

## Part 1: Cross-Encoder Model Skeleton

Below is a skeleton of the `CrossEncoderModel` class. You will fill in the `TODO` sections with your own code and explanations.

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class CrossEncoderModel(nn.Module):
    def __init__(self, model_name, max_num_labels):
        """
        Args:
            model_name (str): Name of the pretrained model (e.g., "bert-base-uncased").
            max_num_labels (int): Maximum number of labels that any text can have.
        """
        super(CrossEncoderModel, self).__init__()
        # TODO: 1) Load a pretrained transformer model and its tokenizer.
        #       2) Initialize a classifier head (nn.Linear) that maps from hidden_size -> 1 (for each label).
        
        # Example:
        # self.encoder = ...
        # self.tokenizer = ...
        # self.classifier = ...
        
        self.max_num_labels = max_num_labels

    def _build_crossencoder_inputs(self, text, labels):
        """
        Build a single cross-encoder input for one example:
            [CLS] label_1 [CLS] label_2 ... [CLS] label_k [SEP] text [SEP]

        Returns:
            input_ids (torch.LongTensor): Token indices for the concatenated sequence.
            attention_mask (torch.LongTensor): Attention mask for that sequence.
            label_cls_positions (torch.LongTensor): Indices of the `[CLS]` tokens corresponding to each label.
        """
        # TODO: 1) Construct a string that inserts "[CLS] " before each label, 
        #          then "[SEP]" + text + "[SEP]" at the end.
        #       2) Tokenize this string without automatically adding special tokens, 
        #          because you are adding them manually.
        #       3) Find the positions where `token_id == self.tokenizer.cls_token_id`
        #          for the label embeddings.
        #       4) Return input_ids, attention_mask, and label_cls_positions.
        
        # Pseudocode:
        # label_part = ""
        # for label in labels:
        #     label_part += f"{self.tokenizer.cls_token} {label} "
        #
        # combined_str = f"{label_part}{self.tokenizer.sep_token} {text} {self.tokenizer.sep_token}"
        #
        # encoding = self.tokenizer(
        #     combined_str,
        #     add_special_tokens=False,
        #     return_tensors="pt"
        # )
        #
        # input_ids = ...
        # attention_mask = ...
        #
        # all_cls_positions = (input_ids == self.tokenizer.cls_token_id).nonzero(as_tuple=True)[0]
        # label_cls_positions = all_cls_positions[:len(labels)]
        #
        # return input_ids, attention_mask, label_cls_positions
        raise NotImplementedError

    def forward(self, texts, batch_labels):
        """
        Args:
            texts (List[str]): List of texts with batch size B.
            batch_labels (List[List[str]]): List of label-lists for each text.

        Returns:
            scores (torch.FloatTensor): Shape [B, max_num_labels], 
                                        containing the predicted relevance for each label.
            mask (torch.BoolTensor): Shape [B, max_num_labels], 
                                     indicating which label positions are valid for each example.
        """
        # TODO: 1) For each example (text + label list), build crossencoder inputs using `_build_crossencoder_inputs`.
        #       2) Use the tokenizer's `.pad(...)` to handle variable-length sequences across the batch.
        #       3) Pass the padded batch through your transformer encoder.
        #       4) Gather the [CLS] embeddings for each label from the `last_hidden_state`.
        #       5) Pad them to [B, max_num_labels, hidden_dim].
        #       6) Pass through a classifier (linear layer) => [B, max_num_labels].
        #       7) Apply sigmoid if doing multi-label classification.
        #       8) Return (scores, mask).
        
        # Step-by-step outline:
        # 1. Prepare lists for input_ids, attention_masks, label_positions_batch.
        # 2. For i in range(len(texts)):
        #       input_ids_i, att_mask_i, label_pos_i = self._build_crossencoder_inputs(texts[i], batch_labels[i])
        #       store them in lists
        #
        # 3. Use tokenizer.pad(...) to get padded input_ids and attention_mask for the entire batch.
        #
        # 4. self.encoder(...) => get last_hidden_state (B, seq_len, hidden_dim).
        #
        # 5. For each example i, gather the embeddings from the label positions.
        # 6. Construct a zero tensor padded_label_embeddings (B, max_num_labels, hidden_dim).
        #    Construct a boolean mask (B, max_num_labels) to mark real vs. padded labels.
        #
        # 7. Run classifier => logits => apply sigmoid => scores.
        #
        # return (scores, mask)
        raise NotImplementedError

    @torch.no_grad()
    def forward_predict(self, texts, labels):
        """
        Args:
            texts (List[str]): List of input texts.
            labels (List[List[str]]): List of labels corresponding to each text.

        Returns:
            A list of dictionaries, each containing:
               {
                   "text": str,
                   "scores": { label: float_score, ... }
               }
        """
        # TODO: 1) Call self.forward(texts, labels) to get scores and mask.
        #       2) Build a result dict for each text, enumerating each label that is valid in the mask.
        #       3) Convert model outputs to float with appropriate formatting.
        #       4) Return the list of result dictionaries.
        raise NotImplementedError
```

### Your Tasks

1. **Complete the Constructor**  
   - Load a pretrained `AutoModel` and `AutoTokenizer` from `model_name`.  
   - Create a classifier head (`nn.Linear`) that takes in the hidden dimension (e.g., `self.encoder.config.hidden_size`) and outputs a single score.

2. **Implement `_build_crossencoder_inputs`**  
   - Concatenate the labels and text with the special tokens (multiple `[CLS]`, then `[SEP] text [SEP]`).  
   - Tokenize with `add_special_tokens=False`.  
   - Identify the positions of each label’s `[CLS]` token.  

3. **Implement `forward`**  
   - Convert a batch of `(texts, labels)` into padded `input_ids` and `attention_mask`.  
   - Pass them through the encoder.  
   - Gather `[CLS]` embeddings for each label.  
   - Pass those embeddings through your classifier to get scores.  
   - Build a `mask` to indicate valid (real) vs. padded labels.  
   - Return `(scores, mask)`.  

4. **Implement `forward_predict`**  
   - Use `forward` to get `scores` and `mask`.  
   - Build a JSON-like output for each text, where `{"text": ..., "scores": {...}}`.  

---

## Part 2: Test Your Implementation

After filling in your code, run a quick test:

```python
if __name__ == "__main__":
    model_name = "bert-base-uncased"
    max_num_labels = 5

    model = CrossEncoderModel(model_name, max_num_labels)

    texts = ["I love machine learning.", "Deep learning models are powerful."]
    batch_labels = [
        ["AI", "Machine Learning"],
        ["Deep Learning", "Neural Networks", "AI"]
    ]

    scores, mask = model(texts, batch_labels)
    print("Scores:", scores)
    print("Mask:", mask)

    predictions = model.forward_predict(texts, batch_labels)
    print("Predictions:", predictions)
```

Check that:
1. `scores` has shape `[B, max_num_labels]` (`B` = number of input texts).  
2. `mask` is boolean with the same shape.  
3. The final JSON-like outputs in `predictions` are as expected.

---

## Part 3: Discussion

1. **Scalability**  
   - Why might a Cross-Encoder become computationally expensive if you have many labels?  
   - Contrast this with the Bi-Encoder approach.

2. **Representation**  
   - In a Cross-Encoder, each label’s representation is conditioned on the text *and* the other labels.  
   - Discuss how this might help performance compared to a Bi-Encoder.

3. **Future Extensions** (Bonus)
   - Could you modify the classifier to produce multi-class probabilities instead of multi-label?  
   - Could you incorporate attention pooling over the text, rather than focusing on `[CLS]` tokens?

Prepare short answers to these questions along with your code.
