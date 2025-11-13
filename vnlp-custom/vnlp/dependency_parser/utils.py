# File: /home/ben/miniconda3/envs/bookanalysis/lib/python3.12/site-packages/vnlp/dependency_parser/utils.py

import numpy as np

def dp_pos_to_displacy_format(dp_result, pos_result=None):
    """
    Converts dependency parsing and part-of-speech results to the
    dictionary format required by the spacy.displacy visualization tool.
    This version is based on the robust logic from the original library.
    """
    if pos_result is None:
        pos_result = [(res[1], '') for res in dp_result]

    words_data = [{'text': token, 'tag': pos_tag} for token, (__, pos_tag) in zip(dp_result, pos_result)]

    arcs_data = []
    for word_idx, _, head_idx, dep_label in dp_result:
        # Root is indicated by head_idx 0; displacy handles this by self-loops or omitting.
        if head_idx <= 0:
            continue
        
        # displacy uses 0-based indexing
        start = word_idx - 1
        end = head_idx - 1
        direction = 'left' if start > end else 'right'
        
        # Ensure start is always less than end for displacy arc rendering
        if direction == 'left':
            start, end = end, start
            
        arcs_data.append({'start': start, 'end': end, 'label': dep_label, 'dir': direction})
    
    return [{'words': words_data, 'arcs': arcs_data}]

def decode_arc_label_vector(vector, sentence_max_len, label_vocab_size):
    """
    Decodes the model's raw output vector into a predicted arc and label.
    This uses the exact slicing from the original library for correctness.
    """
    # First part of the vector corresponds to the arc prediction (+1 for root)
    arc_logits = vector[:sentence_max_len + 1]
    arc = np.argmax(arc_logits, axis=-1)

    # Second part corresponds to the label prediction (+1 for padding/unknown)
    label_start_index = sentence_max_len + 1
    label_end_index = label_start_index + label_vocab_size + 1
    label_logits = vector[label_start_index:label_end_index]
    label = np.argmax(label_logits, axis=-1)
    
    return arc, label