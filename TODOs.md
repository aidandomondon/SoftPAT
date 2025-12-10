### System Generation Prompt

```
We are trying to create an approach using the attached paper as a reference. This replaces the need for discretized attack and defense hard prompts with soft prompts in the embedding space. Create a system that is implements the following approach below:

- Read in harmful data from harmful data directory and read in benign data from benign data directory
- Instantiate LLM
- **GOAL** In alternating steps, optimize two soft prompts– defensive and offensive –meant to respectively discourage and encourage the LLM answering harmful goals.
  - Training details:
    - Defensive:
      - Goal: encourage LLM to refuse responding to harmful goals from harmful behaviors dataset while retaining usefulness in responding to benign goals from benign dataset.
      - Training details:
        - Loss function: (1 - alpha) * Loss_{harmful} + (alpha) * Loss_{benign} where alpha is a balancing parameter.
          - Loss_{harmful} evaluated by using the defensive and offensive soft prompts on the harmful behaviors dataset with the following assembly structure: "<defensive soft prompt> <harmful goal (embedded)> <offensive soft prompt>"
          - Loss_{benign} evaluated by using the defensive soft prompt on the benign behaviors dataset with the following assembly structure: "<defensive soft prompt> <benign goal (embedded)>". Note the absence of an offensive soft prompt.
    - Offensive:
      - Goal: encourage LLM to answer harmful goals from harmful behaviors dataset
      - Training details:
        - Loss evaluated by using the defensive and offensive soft prompts on the harmful behaviors dataset with the following assembly structure: "<defensive soft prompt> <harmful goal (embedded)> <offensive soft prompt>"
  - Evaluation:
    - We evaluate the success of this approach by calculating the attack success rate (ASR) of harmful goals using the harmful dataset. Ensure that the data samples used to evaluate have no overlap with the data samples used to train
- Further details:
  - Instantiate the soft prompts randomly
  - Run the optimization steps for a variable number of iterations T.
  - Parameterize the number of data samples used for training and test so they are easily customizable.

Make a script for running this experimental approach.
```
