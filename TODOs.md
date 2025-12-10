## TODOs

- :) Get baseline PAT working w/ existing readme steps
  - :) get env working
  - :) highlight important files to change
  - :) understand how:
    - :) gradients are calc'd
    - :) how attack, def prompts are initialized
- :) choose model: Vicuna
- implment soft prompting
- outline what experiments we want done for weds presentation:
  - comparison wiht advbench
  - table 1
  - fig 2
  - fig 4a (optional b if we have time)

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
