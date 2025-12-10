## TODOs

- :) Get baseline PAT working w/ existing readme steps
  - :) get env working
  - :) highlight important files to change
  - :) understand how:
    - :) gradients are calc'd
    - :) how attack, def prompts are initialized
- :) choose model: Vicuna
- :) implment soft prompting
- :) outline what experiments we want done for weds presentation:
  - comparison wiht advbench

NOTE: `run_gcg_attack.sh` is overwritten to work for soft prompts

Trace for prompt training
run_gcg -> main_soft.py -> instantiates attack (ProgressiveMultiPromptAttack) -> attack.run -> instantiates MultiPromptAttack and runs this -> MultiPromptAttack init references prompt manager -> SoftpromptMultiPromptAttack -> step and defense step

Investigating control template:

- Every prompt manager references an attack prompt
- attack prompt uses conv_template

How do we integrate soft prompts into system?

- Need to determine how to insert attack + defense embedded prompts into embedded inputs
- Is eval the same?
  - same process as training, embed inputs THEN append attack + defense
- ignore control in conv_template?

Actual TODOs:

- setting self.control does handle all passed controls in ProgressiveMultiPromptAttack
- self.control_str BEWARE!! (somewhere)
- spooky attack.test_all() -> attack is of type MultiPromptAttack
- control template blah

- can embed in step and defense step of SoftpromptMultiPromptAttack

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

Results:

- 5 iters, 25 of each prompt, ASR with= 98%, ASR w/o = 60%
- 10 iters, 25 of each prompt, ASR with= 96%, ASR w/o = 56%
- 15 iters, 25 of each prompt, ASR with= 92%, ASR w/o = 56%
- 20 iters, 25 of each prompt, ASR with= 92%, ASR w/o = 54%
- 50 iters, 25 of each prompt, ASR with= 91%, ASR w/o = 51%
