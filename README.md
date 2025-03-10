# Atom of Thought (AoT) for LangChain

A LangChain implementation of the Atom of Thought (AoT) prompting technique, a meta-reasoning framework that enhances LLMs' problem-solving capabilities by combining multiple solution strategies.

## Features

- **Multiple Reasoning Strategies**: Direct, Decompose, and Contract
- **Ensemble Decision Making**: Selects the best solution approach based on consensus
- **Recursive Processing**: Can process complex problems recursively
- **Dependency Tracking**: Handles dependent sub-questions and calculates depth
- **Seamless LangChain Integration**: Works with any LangChain chat model

## Installation

```bash
# Using pip
pip install atom-of-thought

# From source
git clone https://github.com/benman1/atom-of-thought.git
cd atom-of-thought
pip install -e .
```

## Quick Start

```python
from langchain_openai import ChatOpenAI
from atom_of_thought.aot import AtomOfThought
import asyncio

async def main():
    # Initialize with your LangChain model
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0
    )
    
    # Create AoT instance
    aot = AtomOfThought(llm=llm)
    
    # Sample question
    question = "If a rectangle has a length of 10 units and a width of 5 units, what is its area in square units?"
    
    # Solve using AoT
    result = await aot.solve(question)
    
    # Print results
    print(f"Best method: {result['method']}")
    print(f"Answer: {result['answer']}")
    print(f"Scores: {result['scores']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## How It Works

AoT improves LLM reasoning by:

1. **Direct Approach**: Solves the problem in one attempt
2. **Decomposition**: Breaks the problem into sub-questions
3. **Contraction**: Optimizes the question based on decomposition
4. **Ensemble**: Combines results to select the best answer

This approach is particularly effective for:
- Multi-step reasoning problems
- Math word problems
- Multiple-choice questions
- Multi-hop reasoning tasks

## Examples

### Multi-step Math Problem

```python
from atom_of_thought.aot import AtomOfThought
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)
aot = AtomOfThought(llm=llm)
question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

result = aot.solve(question)
print(result["answer"])  # Output: 18
```

### Multiple-Choice Question

```python
question = "How can a short circuit be detected?\nA: a meggar\nB: an ammeter\nC: an ohmmeter\nD: an oscillscope"

result = aot.solve(question)
print(result["answer"])  # Output: C
```

## Advanced Usage

### Custom System Prompts

```python
system_prompt = "You are a math professor who gives step-by-step solutions."
result = aot.solve(question, system_prompt=system_prompt)
```

### Adjusting Recursion Depth

```python
# Create AoT with custom depth
aot = AtomOfThought(llm=llm, atom_depth=2)

# Or specify depth per question
result = aot.solve(question, depth=1)
```

## References

This implementation is based on the original Atom of Thought paper. If you use this in your research, please cite:

```
@article{atomofthought2023,
  title={Atom of Thought: Enhancing LLM Reasoning via Atomized Thinking},
  author={Original Authors},
  journal={arXiv preprint},
  year={2023}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
