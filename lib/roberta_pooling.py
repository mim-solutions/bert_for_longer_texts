import torch

# Functions for preparing input for longer texts - based on https://www.kdnuggets.com/2021/04/apply-transformers-any-length-text.html

def tokenize_all_text(text,tokenizer):
    '''
    Tokenizes the entire text without truncation and without special tokens
    
    Parameters:
    text - single str with arbitrary length
    tokenizer - object of class transformers.PreTrainedTokenizerFast

    Returns:
    tokens - dictionary of the form
    {
    'input_ids' : [...]
    'token_type_ids' : [...]
    'attention_mask' : [...]
    }
    '''
    tokens = tokenizer.encode_plus(text, add_special_tokens=False,
                               return_tensors='pt')
    return tokens

def split_overlapping(array,size,step,minimal_length):
    ''' Helper function for dividing arrays into overlapping chunks '''
    result = [array[i:i+size] for i in range(0,len(array),step)]
    if len(result) > 1:
        result = [x for x in result if len(x) >= minimal_length] # ignore chunks with less then minimal_length number of tokens 
    return result

def split_tokens_into_smaller_chunks(tokens, size, step, minimal_length):
    ''' Splits tokens into overlapping chunks with given size and step'''
    assert size <= 510
    input_id_chunks = split_overlapping(tokens['input_ids'][0],size,step,minimal_length)
    mask_chunks = split_overlapping(tokens['attention_mask'][0],size,step,minimal_length)
    return input_id_chunks, mask_chunks

def add_special_tokens_at_beginning_and_end(input_id_chunks,mask_chunks):
    ''' Adds special CLS token (token id = 101) at the beginning and SEP token (token id = 102) at the end of each chunk'''
    for i in range(len(input_id_chunks)):
        # adding CLS (token id 101) and SEP (token id 102) tokens
        input_id_chunks[i] = torch.cat([torch.Tensor([101]),input_id_chunks[i],torch.Tensor([102])])
        mask_chunks[i] = torch.cat([torch.Tensor([1]),mask_chunks[i],torch.Tensor([1])])
        
def add_padding_tokens(input_id_chunks,mask_chunks):
    ''' Adds padding tokens (token id = 0) at the end to make sure that all chunks have exactly 512 tokens '''
    for i in range(len(input_id_chunks)):
        # get required padding length
        pad_len = 512 - input_id_chunks[i].shape[0]
        # check if tensor length satisfies required chunk size
        if pad_len > 0:
            # if padding length is more than 0, we must add padding
            input_id_chunks[i] = torch.cat([
                input_id_chunks[i], torch.Tensor([0] * pad_len)
            ])
            mask_chunks[i] = torch.cat([
                mask_chunks[i], torch.Tensor([0] * pad_len)
            ])
            
def stack_tokens_from_all_chunks(input_id_chunks,mask_chunks):
    ''' Reshapes data to a form compatible with roberta model input'''
    input_ids = torch.stack(input_id_chunks)
    attention_mask = torch.stack(mask_chunks)

    return input_ids.long(), attention_mask.int()

def transform_text_to_model_input(text,tokenizer, size = 510, step = 510, minimal_length = 100):
    ''' Transforms the entire text to model input of roberta model'''
    tokens = tokenize_all_text(text,tokenizer)
    input_id_chunks, mask_chunks = split_tokens_into_smaller_chunks(tokens,size,step,minimal_length)
    add_special_tokens_at_beginning_and_end(input_id_chunks,mask_chunks)
    add_padding_tokens(input_id_chunks,mask_chunks)
    input_ids, attention_mask = stack_tokens_from_all_chunks(input_id_chunks,mask_chunks)
    return [input_ids, attention_mask]
