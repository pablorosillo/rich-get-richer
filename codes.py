import random
import numpy as np
from scipy.special import zeta
from collections import Counter

def calculate_rho_custom(alpha, N_t):
    """
    Calculates the dynamic innovation probability (rho) for the generalized model.
    
    Parameters:
    - alpha (float): The parameter of the model.
    - N_t (int): The current number of unique words (types).
    
    Returns:
    - float: The probability of introducing a new word.
    """
    N_eff = max(N_t, 0)
    numerator = 1.0 - alpha
    term = alpha * (1.0 - alpha) * zeta(alpha) * ((N_eff + 1)**(alpha - 1.0))
    denominator = 1.0 + term
    
    if denominator <= 0: 
        return 0.0
        
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def simulate_simons_model(num_tokens, rho):
    """
    Simulates Simon's model with a constant innovation probability.
    
    Parameters:
    - num_tokens (int): Total number of words to generate.
    - rho (float): Constant probability of introducing a new word.
    
    Returns:
    - current_type_id (int): Total number of unique words generated.
    - frequencies (Counter): Frequency distribution of the words.
    """
    frequencies = Counter()
    tokens = []
    current_type_id = 0
    
    for _ in range(num_tokens):
        # 'not tokens' ensures the very first word is always an innovation
        if not tokens or random.random() < rho:
            # Innovation: introduce a new word
            current_type_id += 1
            word = current_type_id
        else:
            # Preferential attachment: pick an existing word
            word = random.choice(tokens)
            
        tokens.append(word)
        frequencies[word] += 1
        
    return current_type_id, frequencies


def simulate_generalized_model(num_tokens, alpha):
    """
    Simulates the generalized model with a dynamic innovation probability
    based on the theoretical custom rate.
    
    Parameters:
    - num_tokens (int): Total number of words to generate.
    - alpha (float): The parameter of the model, used to calculate dynamic rho.
    
    Returns:
    - current_type_id (int): Total number of unique words generated.
    - frequencies (Counter): Frequency distribution of the words.
    """
    frequencies = Counter()
    tokens = []
    current_type_id = 0
    
    for _ in range(num_tokens):
        # Calculate dynamic probability of innovation based on current vocabulary size
        rho = calculate_rho_custom(alpha, current_type_id)
        
        # 'not tokens' ensures the very first word is always an innovation
        if not tokens or random.random() < rho:
            # Innovation: introduce a new word
            current_type_id += 1
            word = current_type_id
        else:
            # Preferential attachment: pick an existing word
            word = random.choice(tokens)
            
        tokens.append(word)
        frequencies[word] += 1
        
    return current_type_id, frequencies


if __name__ == "__main__":
    # --- Example Usage ---
    N_TOKENS = 5000
    ALPHA = 0.9
    
    # In Simon's model
    CONSTANT_RHO = 0.1 

    print("Running Simon's Model...")
    simon_types, simon_freqs = simulate_simons_model(N_TOKENS, CONSTANT_RHO)
    print(f"Simon's Model -> Tokens: {N_TOKENS}, Types: {simon_types}")

    print("\nRunning Generalized Model...")
    gen_types, gen_freqs = simulate_generalized_model(N_TOKENS, ALPHA)
    print(f"Generalized Model -> Tokens: {N_TOKENS}, Types: {gen_types}")
