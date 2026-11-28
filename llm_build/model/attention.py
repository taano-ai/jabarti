# Input: 3 tokens, 2 features each -> (3, 2)
x = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])

# Weight matrices: 3 matrices shared by all tokens, each (d_model, d_model) = (2, 2)
q_w = torch.tensor([[2.0, 0.0],
                    [1.0, 3.0]])

k_w = torch.tensor([[2.0, 1.0],
                    [0.0, 4.0]])

v_w = torch.tensor([[2.0, 5.0],
                    [0.0, 3.0]])

# STEP 1: build q, k, v from the same x (one row per token) -> each (3, 2)
q = x @ q_w   # what each token is looking for
k = x @ k_w   # what each token offers for matching
v = x @ v_w   # content each token hands over (not used until step 4)

# STEP 2: scores = every query vs every key (dot products) -> (3, 3)
# scores[i][j] = how much token i's query matches token j's key
scores = q @ k.T

# STEP 3 (next): softmax on each row -> weights that are positive and sum to 1
# STEP 4 (after): output = weights @ v -> (3, 2), same shape as x
scores