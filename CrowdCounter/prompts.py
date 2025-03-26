counterspeech_prompt = """Counterspeech is a strategic response to hate speech, aiming to foster understanding or discourage harmful behavior. 
                          A good counterspeech to this hate speech - "{hate_speech}" is: """

# counterspeech_prompt = """Write a good counterspeech to this hate speech - "{hate_speech}" is: """

counterspeech_prompt_llama_two = """[INST] <<SYS>>  You are an helpful agent who generates a specific type of counterspeech to the hate speech provided by the user. 

Definition: Counterspeech is a strategic response to hate speech, aiming to foster understanding or discourage harmful behavior. <</SYS>>

{hate_speech} [/INST]"""



counterspeech_prompt_llama_three = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an helpful agent who generates a specific type of counterspeech to the hate speech provided by the user. 

Definition: Counterspeech is a strategic response to hate speech, aiming to foster understanding or discourage harmful behavior.

<|eot_id|><|start_header_id|>user<|end_header_id|>{hate_speech}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""




type_specific_generation_prompt = """Counterspeech is a strategic response to hate speech, aiming to foster understanding or discourage harmful behavior. Different types of counterspeech include:

1. contradiction: Point out any inconsistencies or contradictions in the hate speech. Explain and rationalize past behavior or encourage the individual to reflect on their statements.

2. empathy affiliation: Respond with a friendly, empathetic, or peaceful tone to counteract hostility or violence in the original message.

3. humour: Utilize humor, ranging from conciliatory to provocative, through forms like caricature and sarcasm to address the hate speech.

4. questions: Pose thoughtful questions to challenge the hate speaker's sources of information or encourage self-reflection.

5. shaming: Leverage societal stigma associated with offensive terms to prompt speakers to reconsider their statements. Provide insight into the reasons behind the hateful nature of the speech.

6. warning of consequences: Emphasize the potential harm caused by hate speech, highlighting its ability to incite real-world actions. Stress the visibility of online content to the speaker's offline network. 

A "{type}" type good counterspeech to this hate speech - {hate_speech} is: """




type_specific_generation_prompt_llama_two = """[INST] <<<SYS>>  You are an helpful agent who generates a counterspeech of type - {type} to the hate speech provided by the user. 

Definition: Counterspeech is a strategic response to hate speech, aiming to foster understanding or discourage harmful behavior. Different types of counterspeech include:

1. contradiction: Point out any inconsistencies or contradictions in the hate speech. Explain and rationalize past behavior or encourage the individual to reflect on their statements.

2. empathy affiliation: Respond with a friendly, empathetic, or peaceful tone to counteract hostility or violence in the original message.

3. humour: Utilize humor, ranging from conciliatory to provocative, through forms like caricature and sarcasm to address the hate speech.

4. questions: Pose thoughtful questions to challenge the hate speaker's sources of information or encourage self-reflection.

5. shaming: Leverage societal stigma associated with offensive terms to prompt speakers to reconsider their statements. Provide insight into the reasons behind the hateful nature of the speech.

6. warning of consequences: Emphasize the potential harm caused by hate speech, highlighting its ability to incite real-world actions. Stress the visibility of online content to the speaker's offline network. 

<</SYS>>
{hate_speech} [/INST] """




type_specific_generation_prompt_llama_three = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an helpful agent who generates a counterspeech of type - {type} to the hate speech provided by the user.

Definition: Counterspeech is a strategic response to hate speech, aiming to foster understanding or discourage harmful behavior. Different types of counterspeech include:

1. contradiction: Point out any inconsistencies or contradictions in the hate speech. Explain and rationalize past behavior or encourage the individual to reflect on their statements.

2. empathy affiliation: Respond with a friendly, empathetic, or peaceful tone to counteract hostility or violence in the original message.

3. humour: Utilize humor, ranging from conciliatory to provocative, through forms like caricature and sarcasm to address the hate speech.

4. questions: Pose thoughtful questions to challenge the hate speaker's sources of information or encourage self-reflection.

5. shaming: Leverage societal stigma associated with offensive terms to prompt speakers to reconsider their statements. Provide insight into the reasons behind the hateful nature of the speech.

6. warning of consequences: Emphasize the potential harm caused by hate speech, highlighting its ability to incite real-world actions. Stress the visibility of online content to the speaker's offline network. 

<|eot_id|><|start_header_id|>user<|end_header_id|> {hate_speech} <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""




























############################ These are classification prompts
type_specific_classification_prompt = """Counterspeech is a strategic response to hate speech, aiming to foster understanding or discourage harmful behavior. Different types of counterspeech include:

1. contradiction: Point out any inconsistencies or contradictions in the hate speech. Explain and rationalize past behavior or encourage the individual to reflect on their statements.

2. empathy affiliation: Respond with a friendly, empathetic, or peaceful tone to counteract hostility or violence in the original message.

3. humour: Utilize humor, ranging from conciliatory to provocative, through forms like caricature and sarcasm to address the hate speech.

4. questions: Pose thoughtful questions to challenge the hate speaker's sources of information or encourage self-reflection.

5. shaming: Leverage societal stigma associated with offensive terms to prompt speakers to reconsider their statements. Provide insight into the reasons behind the hateful nature of the speech.

6. warning of consequences: Emphasize the potential harm caused by hate speech, highlighting its ability to incite real-world actions. Stress the visibility of online content to the speaker's offline network.

Given this hate speech - "{hate_speech}" and counterspeech - "{counter_speech}" what are the types present in the counterspeech out of the ones listed ? Give in the format of a list """



type_specific_classification_prompt_wo_hs = """Counterspeech is a strategic response to hate speech, aiming to foster understanding or discourage harmful behavior. Different types of counterspeech include:

1. contradiction: Point out any inconsistencies or contradictions in the hate speech. Explain and rationalize past behavior or encourage the individual to reflect on their statements.

2. empathy affiliation: Respond with a friendly, empathetic, or peaceful tone to counteract hostility or violence in the original message.

3. humour: Utilize humor, ranging from conciliatory to provocative, through forms like caricature and sarcasm to address the hate speech.

4. questions: Pose thoughtful questions to challenge the hate speaker's sources of information or encourage self-reflection.

5. shaming: Leverage societal stigma associated with offensive terms to prompt speakers to reconsider their statements. Provide insight into the reasons behind the hateful nature of the speech.

6. warning of consequences: Emphasize the potential harm caused by hate speech, highlighting its ability to incite real-world actions. Stress the visibility of online content to the speaker's offline network.

Given this counterspeech - "{counter_speech}" what are the types present in the counterspeech out of the ones listed ? Give in the format of a list"""




type_specific_classification_prompt_wo_hs_fewshot = """Counterspeech is a strategic response to hate speech, aiming to foster understanding or discourage harmful behavior. Different types of counterspeech include:

1. contradiction: Point out any inconsistencies or contradictions in the hate speech. Explain and rationalize past behavior or encourage the individual to reflect on their statements.

2. empathy affiliation: Respond with a friendly, empathetic, or peaceful tone to counteract hostility or violence in the original message.

3. humour: Utilize humor, ranging from conciliatory to provocative, through forms like caricature and sarcasm to address the hate speech.

4. questions: Pose thoughtful questions to challenge the hate speaker's sources of information or encourage self-reflection.

5. shaming: Leverage societal stigma associated with offensive terms to prompt speakers to reconsider their statements. Provide insight into the reasons behind the hateful nature of the speech.

6. warning of consequences: Emphasize the potential harm caused by hate speech, highlighting its ability to incite real-world actions. Stress the visibility of online content to the speaker's offline network.

Task is to classify the type of counterspeech given the counterspeech. Some examples are

{examples}

Given this counterspeech - "{counter_speech}" what are the types present in the counterspeech out of the ones listed ? Give in the format of a list"""