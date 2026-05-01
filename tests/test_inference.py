from inference.generate import generate, promptFormat

def test_generate_basic():
    prompt = promptFormat("Write a factorial function", "5")
    output = generate(prompt)

    assert isinstance(output, str)
    assert len(output) > 0

def test_no_eos_token():
    prompt = promptFormat("Write a factorial function", "5")
    output = generate(prompt)

    assert "<|endoftext|>" not in output

def test_generate_empty_input():
    prompt = promptFormat("Write factorial")
    output = generate(prompt)

    assert isinstance(output, str)

