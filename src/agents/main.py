#!/usr/bin/env python
import datetime
import sys
from agents.crew import AgentsCrew
import litellm

litellm.verbose=True

def run():
    """
    Run the crew.
    """
    inputs = {
        'current_date': datetime.datetime.now().strftime("%Y-%m-%d"),
        'instagram_description': input('Enter the page description here: '),
        'topic_of_the_week': input('Enter the topic of the week here: '),
    }
    AgentsCrew().crew().kickoff(inputs=inputs)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        AgentsCrew().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        AgentsCrew().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        AgentsCrew().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")
