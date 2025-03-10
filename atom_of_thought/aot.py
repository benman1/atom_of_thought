"""Atom of Thought (AoT) implementation."""
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.output_parsers import XMLOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
import json
import re

class SubQuestion(BaseModel):
    """Schema for a sub-question in the problem decomposition."""
    description: str = Field(description="The sub-question text")
    answer: str = Field(description="The answer to the sub-question")
    depend: List[int] = Field(description="Indices of sub-questions this depends on", default_factory=list)

class DecompositionResult(BaseModel):
    """Schema for the result of problem decomposition."""
    sub_questions: List[SubQuestion] = Field(description="List of sub-questions")
    answer: str = Field(description="The final answer to the original question")

class AtomOfThought:
    """
    Implements the Atom of Thought (AoT) reasoning technique in LangChain.

    This class provides multiple reasoning strategies (direct, decompose, contract)
    and combines them using ensemble techniques to select the best approach.

    Attributes:
        llm: Language model to use for generating responses
        max_retries: Maximum number of retries for failed attempts
        atom_depth: Maximum recursion depth for the algorithm
    """

    def __init__(self, llm=None, max_retries=2, atom_depth=3):
        """
        Initialize AtomOfThought with a language model.

        Args:
            llm: Language model to use (defaults to ChatOpenAI)
            max_retries: Maximum number of retries for failed attempts
            atom_depth: Maximum recursion depth for the algorithm
        """
        # Use provided LLM or create default
        self.llm = llm or ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_retries=max_retries
        )
        self.max_retries = max_retries
        self.atom_depth = atom_depth

        # Initialize parsers
        self.xml_parser = XMLOutputParser()
        self.json_parser = JsonOutputParser(pydantic_object=DecompositionResult)

    def calculate_depth(self, sub_questions: List[Dict[str, Any]]) -> int:
        """
        Calculate the maximum dependency depth of sub-questions.

        Uses the Floyd-Warshall algorithm to find the longest dependency chain.

        Args:
            sub_questions: List of sub-question dictionaries with "depend" field

        Returns:
            int: Maximum dependency depth
        """
        try:
            n = len(sub_questions)
            # Initialize distances matrix with infinity
            distances = [[float("inf")] * n for _ in range(n)]
            # Set direct dependencies
            for i, sub_q in enumerate(sub_questions):
                # Distance to self is 0
                distances[i][i] = 0
                # Set direct dependencies with distance 1
                for dep in sub_q.get("depend", []):
                    distances[dep][i] = 1
            # Floyd-Warshall algorithm to find shortest paths
            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        if distances[i][k] != float("inf") and distances[k][j] != float("inf"):
                            distances[i][j] = min(
                                distances[i][j], distances[i][k] + distances[k][j]
                            )
            # Find maximum finite distance
            max_depth = 0
            for i in range(n):
                for j in range(n):
                    if distances[i][j] != float("inf"):
                        max_depth = max(max_depth, distances[i][j])
            return int(max_depth)
        except:
            return 3

    async def direct_solve(self, question: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Direct solution approach - attempt to solve the problem in one go.

        Args:
            question: The question to solve
            system_prompt: Optional system prompt to guide the model

        Returns:
            Dict containing the response and answer
        """
        default_system = "You are a helpful assistant. Answer the question directly and concisely."
        system_prompt = system_prompt or default_system

        # Create template with XML output instructions
        template = """Question: {question}

Answer the question directly and concisely.
Enclose your final answer within <answer></answer> tags."""

        prompt = PromptTemplate(
            template=template,
            input_variables=["question"],
        )

        # Create chain with XML parser for the answer
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt.format(question=question))
        ]

        # Invoke with retry logic
        for attempt in range(self.max_retries):
            try:
                response = self.llm.invoke(messages).content

                # Extract the answer using the XML parser
                try:
                    answer_xml = f"<root>{response}</root>"
                    parsed = self.xml_parser.parse(answer_xml)
                    answer = parsed.get("answer", "")
                except:
                    # Fallback to regex if parser fails
                    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
                    answer = answer_match.group(1).strip() if answer_match else ""

                if answer:
                    return {
                        "response": response,
                        "answer": answer
                    }
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return {
                        "response": str(e),
                        "answer": ""
                    }

        # If all attempts fail or no answer found
        return {
            "response": response if 'response' in locals() else "",
            "answer": ""
        }

    async def decompose_solve(self, question: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Decomposition approach - break problem into sub-questions.

        Args:
            question: The question to solve
            system_prompt: Optional system prompt to guide the model

        Returns:
            Dict containing the response, answer, and sub-questions
        """
        default_system = """You are a helpful assistant that breaks down complex problems into steps.
First, decompose the question into sub-questions.
Then, answer each sub-question.
Finally, use those answers to solve the original question."""

        system_prompt = system_prompt or default_system

        # Create template with JSON output format instructions
        template = """Question: {question}

Please decompose this into sub-questions and solve them step by step.
{format_instructions}"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["question"],
            partial_variables={"format_instructions": self.json_parser.get_format_instructions()},
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt.format(question=question))
        ]

        # Invoke with retry logic
        for attempt in range(self.max_retries):
            try:
                response = self.llm.invoke(messages).content

                # Try to parse the JSON response
                try:
                    # First try using the json parser
                    result = self.json_parser.parse(response)
                    # Convert to dict if it's a Pydantic model
                    if hasattr(result, "model_dump"):
                        result = result.model_dump()
                except:
                    # Fallback to regex extraction
                    json_match = re.search(r"\{.*\}", response, re.DOTALL)
                    if json_match:
                        try:
                            result = json.loads(json_match.group(0))
                        except:
                            continue
                    else:
                        continue

                # Validate the result has the expected structure
                if "sub_questions" in result and "answer" in result:
                    # Standardize the keys to match original implementation
                    if "sub_questions" in result:
                        result["sub-questions"] = result.pop("sub_questions")

                    # Calculate depth to verify structure
                    self.calculate_depth(result["sub-questions"])
                    result["response"] = response
                    return result
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return {
                        "response": str(e),
                        "answer": "",
                        "sub-questions": []
                    }

        # If all attempts fail
        return {
            "response": response if 'response' in locals() else "",
            "answer": "",
            "sub-questions": []
        }

    async def contract_solve(
            self,
            question: str,
            decompose_result: Dict[str, Any],
            system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Contraction approach - optimize the question based on decomposition.

        Args:
            question: The original question
            decompose_result: Result from the decomposition step
            system_prompt: Optional system prompt to guide the model

        Returns:
            Dict containing the response, answer, and optimized question
        """
        default_system = """You are a helpful assistant that optimizes questions based on previous decomposition.
Your task is to create a simplified, more direct version of the original question."""

        system_prompt = system_prompt or default_system

        # Format sub-questions for the prompt
        sub_questions_text = "\n".join([
            f"{i+1}. {sq['description']} Answer: {sq['answer']}"
            for i, sq in enumerate(decompose_result.get("sub-questions", []))
        ])

        template = """Original Question: {question}

Here's how the question was decomposed and answered:
{sub_questions}

Based on this decomposition, create an optimized version of the original question that is more direct and easier to answer.
Then answer this optimized question.

Enclose the optimized question within <question></question> tags.
Enclose your final answer within <answer></answer> tags."""

        prompt = PromptTemplate(
            template=template,
            input_variables=["question", "sub_questions"],
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt.format(
                question=question,
                sub_questions=sub_questions_text
            ))
        ]

        # Invoke with retry logic
        for attempt in range(self.max_retries):
            try:
                response = self.llm.invoke(messages).content

                # Extract the question and answer using regex
                optimized_question_match = re.search(r"<question>(.*?)</question>", response, re.DOTALL)
                answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

                optimized_question = ""
                if optimized_question_match:
                    optimized_question = optimized_question_match.group(1).strip()

                answer = ""
                if answer_match:
                    answer = answer_match.group(1).strip()

                # If we got an optimized question but no answer, get a direct solution to it
                if optimized_question and not answer:
                    direct_result = await self.direct_solve(optimized_question, system_prompt)
                    answer = direct_result["answer"]
                    response += "\n\n" + direct_result["response"]

                return {
                    "response": response,
                    "answer": answer,
                    "optimized_question": optimized_question
                }
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return {
                        "response": str(e),
                        "answer": "",
                        "optimized_question": ""
                    }

        # If all attempts fail
        return {
            "response": response if 'response' in locals() else "",
            "answer": "",
            "optimized_question": ""
        }

    async def ensemble_solve(
            self,
            question: str,
            results: List[str],
            system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ensemble approach - combine multiple solutions to get consensus.

        Args:
            question: The original question
            results: List of response strings from different approaches
            system_prompt: Optional system prompt to guide the model

        Returns:
            Dict containing the response and consensus answer
        """
        default_system = """You are a helpful assistant that analyzes multiple answers to select the best one.
Your task is to consider different approaches to solving a problem and determine the most accurate answer."""

        system_prompt = system_prompt or default_system

        responses_text = "\n\n".join([
            f"Approach {i+1}:\n{result}"
            for i, result in enumerate(results)
        ])

        template = """Question: {question}

Here are different approaches to answering this question:
{approaches}

Based on these approaches, what is the most accurate answer?
Provide your final answer within <answer></answer> tags."""

        prompt = PromptTemplate(
            template=template,
            input_variables=["question", "approaches"],
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt.format(
                question=question,
                approaches=responses_text
            ))
        ]

        # Invoke with retry logic
        for attempt in range(self.max_retries):
            try:
                response = self.llm.invoke(messages).content

                # Extract the answer using regex
                answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
                answer = answer_match.group(1).strip() if answer_match else ""

                return {
                    "response": response,
                    "answer": answer
                }
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return {
                        "response": str(e),
                        "answer": ""
                    }

        # If all attempts fail
        return {
            "response": response if 'response' in locals() else "",
            "answer": ""
        }

    def score_answers(self, answer1: str, answer2: str) -> float:
        """
        Score the similarity between two answers.

        Args:
            answer1: First answer string
            answer2: Second answer string

        Returns:
            float: Similarity score between 0 and 1
        """
        if not answer1 or not answer2:
            return 0.0

        # Simple exact match
        if answer1.strip().lower() == answer2.strip().lower():
            return 1.0

        # Try numeric comparison
        try:
            num1 = float(answer1.strip())
            num2 = float(answer2.strip())
            diff = abs(num1 - num2)
            if diff < 0.001:
                return 1.0
            if diff / max(abs(num1), abs(num2)) < 0.01:  # Within 1%
                return 0.9
            return 0.0
        except:
            pass

        # Partial string match
        words1 = set(answer1.lower().split())
        words2 = set(answer2.lower().split())
        if words1 and words2:
            overlap = len(words1.intersection(words2))
            return overlap / max(len(words1), len(words2))

        return 0.0

    async def solve(
            self,
            question: str,
            system_prompt: Optional[str] = None,
            depth: int = None
    ) -> Dict[str, Any]:
        """
        Main method to solve a question using Atom of Thought.

        Args:
            question: The question to solve
            system_prompt: Optional system prompt to guide the model
            depth: Optional recursion depth (defaults to self.atom_depth)

        Returns:
            Dict containing the results, with keys:
                - method: Best method used
                - response: Response from the best method
                - answer: Final answer
                - scores: Scores for each method
                - all_results: Results from all methods
        """
        depth = depth if depth is not None else self.atom_depth

        if depth == 0:
            # Base case: just use direct approach
            direct_result = await self.direct_solve(question, system_prompt)
            return {
                "method": "direct",
                "response": direct_result["response"],
                "answer": direct_result["answer"],
                "scores": {"direct": 1.0},
                "all_results": {"direct": direct_result}
            }

        # Get solutions from different approaches
        direct_result = await self.direct_solve(question, system_prompt)
        decompose_result = await self.decompose_solve(question, system_prompt)

        # Get the optimized question through contraction
        contract_result = await self.contract_solve(question, decompose_result, system_prompt)

        # Ensemble to get consensus
        ensemble_result = await self.ensemble_solve(
            question,
            [direct_result["response"], decompose_result["response"], contract_result["response"]],
            system_prompt
        )

        # Score each result against the ensemble
        scores = [
            self.score_answers(direct_result["answer"], ensemble_result["answer"]),
            self.score_answers(decompose_result["answer"], ensemble_result["answer"]),
            self.score_answers(contract_result["answer"], ensemble_result["answer"])
        ]

        # Select best method
        methods = ["direct", "decompose", "contract", "ensemble"]
        max_score_index = scores.index(max(scores)) if max(scores) > 0 else 3
        best_method = methods[max_score_index]

        results = {
            "direct": direct_result,
            "decompose": decompose_result,
            "contract": contract_result,
            "ensemble": ensemble_result
        }

        # If a deeper recursion is warranted, apply AoT recursively
        if (depth > 1 and
                best_method == "decompose" and
                "sub-questions" in decompose_result and
                len(decompose_result["sub-questions"]) > 0):

            # Calculate actual depth needed
            actual_depth = min(depth - 1, self.calculate_depth(decompose_result["sub-questions"]))

            if actual_depth > 0:
                # Process independent sub-questions recursively
                for i, subq in enumerate(decompose_result["sub-questions"]):
                    if not subq["depend"]:  # Independent question
                        sub_result = await self.solve(
                            subq["description"],
                            system_prompt,
                            depth=actual_depth
                        )
                        # Update the answer if we got a better one
                        if sub_result["answer"]:
                            decompose_result["sub-questions"][i]["answer"] = sub_result["answer"]

        # Return result with metadata
        return {
            "method": best_method,
            "response": results[best_method]["response"],
            "answer": results[best_method]["answer"],
            "scores": dict(zip(methods[:3], scores)),
            "all_results": results,
            "depth": depth
        }

# Example usage
async def main():
    """Example usage of the AtomOfThought class."""
    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0
    )

    # Create AoT instance
    aot = AtomOfThought(llm=llm)

    # Sample question
    question = "If a rectangle has a length of 10 units and a width of 5 units, what is its area in square units?"

    # Optional system prompt to customize behavior
    system_prompt = "You are a helpful math assistant that provides clear, step-by-step solutions."

    # Solve using AoT
    result = await aot.solve(question, system_prompt)

    # Print results
    print(f"Best method: {result['method']}")
    print(f"Answer: {result['answer']}")
    print(f"Scores: {result['scores']}")

if __name__ == "__main__":
    asyncio.run(main())