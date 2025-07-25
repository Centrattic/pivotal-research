def gender_swap_contrast(row):
    """
    Given a row with a name and gender label, return two dicts:
    - original: prompt explicitly states the original gender
    - contrast: prompt explicitly states the swapped gender
    Both dicts have 'prompt', 'target', and 'length' fields.
    """
    name = row['prompt']
    target = row['target']
    gender_word = 'female' if target == 1 else 'male'
    swapped_gender_word = 'male' if gender_word == 'female' else 'female'
    prompt_orig = f"{name} is {gender_word}"
    prompt_contrast = f"{name} is {swapped_gender_word}"
    length_orig = len(prompt_orig)
    length_contrast = len(prompt_contrast)
    original_row = {'prompt': prompt_orig, 'target': target, 'length': length_orig}
    contrast_row = {'prompt': prompt_contrast, 'target': target, 'length': length_contrast}
    return original_row, contrast_row

def language_swap_contrast(row):
    """
    Given a row with a prompt and language label, return two dicts:
    - original: prompt explicitly states the original language
    - contrast: prompt explicitly states the swapped language
    Both dicts have 'prompt', 'target', and 'length' fields.
    Example: original prompt 'hello', target 0 (English)
    -> original: 'hello is in english', contrast: 'hello is in french'
    """
    prompt = row['prompt']
    target = row['target']
    language_word = 'English' if target == 0 else 'French'
    swapped_language_word = 'French' if language_word == 'English' else 'English'
    prompt_orig = f"{prompt} is in {language_word}"
    prompt_contrast = f"{prompt} is in {swapped_language_word}"
    length_orig = len(prompt_orig)
    length_contrast = len(prompt_contrast)
    original_row = {'prompt': prompt_orig, 'target': target, 'length': length_orig}
    contrast_row = {'prompt': prompt_contrast, 'target': target, 'length': length_contrast}
    return original_row, contrast_row
