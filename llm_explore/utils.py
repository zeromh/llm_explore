"""
Utility functions for device configuration, model parameter inspection, prompt generation, 
and model completion for sequence-to-sequence tasks.
"""

import torch

def get_torch_device():
    """
    Determines the appropriate device for computation.

    Returns:
        torch.device: The device to use ('mps' if available, otherwise 'cpu').
    """
    if torch.backends.mps.is_available():
        print("Returned MPS device")
        return torch.device("mps")
    else:
        print("Returned CPU device")
        return torch.device("cpu")

def print_number_of_model_parameters(model):
    """
    Prints and returns the total and trainable parameters of a model.

    Args:
        model (torch.nn.Module): The model to inspect.

    Returns:
        tuple: A tuple containing the total number of parameters and the number of trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Percentage of trainable parameters: {trainable_params / total_params * 100:.2f}%")
    return total_params, trainable_params

def make_n_shot_summary_prompt(example_ids=None, summarize_id=0, data=None, my_set='test'):
    """
    Generates a prompt for summarizing conversations using N-shot examples.

    Args:
        example_ids (list, optional): List of example indices to include in the prompt.
        summarize_id (int): The index of the conversation to summarize.
        data (dict): The dataset containing dialogues and summaries.
        my_set (str): The dataset split to use ('train', 'test', etc.).

    Returns:
        str: The generated prompt.
    """
    prompt = ''
    if example_ids:
        for i in example_ids:
            dialogue = data[my_set]['dialogue'][i]
            human_summary = data[my_set]['summary'][i]
            prompt += f"""
Summarize the following conversation.

{dialogue}

Summary:

{human_summary}
"""
    dialogue = data[my_set]['dialogue'][summarize_id]
    prompt += f"""
Summarize the following conversation.

{dialogue}

Summary:
"""
    return prompt

def get_model_completion(prompt, tokenizer, model, gen_config=None, do_sample=False, max_new_tokens=1000, num_beams=1):
    """
    Generates a model completion for a given prompt.

    Args:
        prompt (str): The input prompt for the model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for encoding and decoding.
        model (transformers.PreTrainedModel): The model to generate the completion.
        gen_config (transformers.GenerationConfig, optional): Configuration for generation.
        do_sample (bool): Whether to sample during generation.
        max_new_tokens (int): Maximum number of new tokens to generate.
        num_beams (int): Number of beams for beam search.

    Returns:
        str: The decoded model output.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentence_encoded = tokenizer(prompt, return_tensors='pt').to(device)
    completion = model.generate(
        input_ids=sentence_encoded.input_ids,
        num_beams=num_beams,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        generation_config=gen_config
    )[0]
    return tokenizer.decode(completion, skip_special_tokens=True)
